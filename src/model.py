from imports import *


# ===============================
# MLP Block 
# ===============================
class MLPBlock(layers.Layer):
    def __init__(self, hidden_dim, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def build(self, input_shape):
        # compute in float32 for stability
        self.dense1 = layers.Dense(self.hidden_dim, activation='gelu', dtype="float32")
        self.dense2 = layers.Dense(self.out_dim, dtype="float32")

    def call(self, x):
        x32 = tf.cast(x, tf.float32)
        x32 = self.dense1(x32)
        x32 = self.dense2(x32)
        # caller decides final dtype; usually I kept it float32 inside the block
        return x32


# ===============================
# Window Partition / Reverse
# ===============================
def window_partition(x, window_size):
    # x: [B, H, W, C]
    B = tf.shape(x)[0]
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    C = tf.shape(x)[3]

    pad_h = (window_size - (H % window_size)) % window_size
    pad_w = (window_size - (W % window_size)) % window_size

    x = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
    Hp = H + pad_h
    Wp = W + pad_w

    x = tf.reshape(
        x, [B, Hp // window_size, window_size, Wp // window_size, window_size, C]
    )
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])  # [B, Nh, Nw, Wh, Ww, C]
    windows = tf.reshape(
        x, [B * (Hp // window_size) * (Wp // window_size), window_size * window_size, C]
    )
    return windows, (H, W, Hp, Wp, pad_h, pad_w)

def window_reverse(windows, window_size, meta):
    # windows: [B * Nh * Nw, window_size*window_size, C]
    H, W, Hp, Wp, pad_h, pad_w = meta
    C = tf.shape(windows)[-1]
    Nh = Hp // window_size
    Nw = Wp // window_size
    B = tf.shape(windows)[0] // (Nh * Nw)

    x = tf.reshape(windows, [B, Nh, Nw, window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])  # [B, Nh, Wh, Nw, Ww, C]
    x = tf.reshape(x, [B, Hp, Wp, C])
    x = x[:, :H, :W, :]
    return x


# ===============================
# Swin Transformer Block
# ===============================
@keras.saving.register_keras_serializable()
class SwinTransformerBlock(layers.Layer):
    def __init__(self, num_heads=4, mlp_ratio=4., window_size=7, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        # keep numerically fragile ops in float32
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, dtype="float32")
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, dtype="float32")
        self.attn = None
        self.mlp = None

    def build(self, input_shape):
        _, H, W, C = input_shape
        self.C = C
        self.attn = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.C // self.num_heads, dtype="float32")
        hidden_dim = int(C * self.mlp_ratio)
        self.mlp = MLPBlock(hidden_dim=hidden_dim, out_dim=C)

    def call(self, x):
        orig_dtype = x.dtype  # usually float16 under mixed precision

        # cast residual to float32 to match the main path
        shortcut = tf.cast(x, tf.float32)

        # norm1 in float32
        x_norm32 = self.norm1(tf.cast(x, tf.float32))

        # window partition (preserves dtype: float32 here)
        windows, meta = window_partition(x_norm32, self.window_size)

        # attention in float32
        x_attn32 = self.attn(windows, windows)

        # reverse windows back to [B,H,W,C] in float32
        x_merged32 = window_reverse(x_attn32, self.window_size, meta)

        # residual add in float32
        x32 = shortcut + x_merged32

        # norm2 + MLP (still float32)
        x32 = self.norm2(x32)
        x32 = self.mlp(x32)  # returns float32
        x32 = x32 + shortcut

        # return to original dtype for the rest of the network
        return tf.cast(x32, orig_dtype)
    
# ===============================
# Patch Operations
# ===============================
def patch_partition(x, patch_size=4, embed_dim=96):
    x = layers.Conv2D(embed_dim, patch_size, patch_size, padding='same')(x)
    return layers.LayerNormalization(epsilon=1e-5)(x)

def patch_merging(x, embed_dim):
    x = layers.Conv2D(embed_dim * 2, 2, 2, padding='same')(x)
    return layers.LayerNormalization(epsilon=1e-5)(x)

def patch_expanding(x, embed_dim):
    x = layers.Conv2DTranspose(embed_dim // 2, 2, 2, padding='same')(x)
    return layers.LayerNormalization(epsilon=1e-5)(x)


# ===============================
# Swin-Unet Architecture
# ===============================
def swin_unet(input_size=(512, 704, 3), num_classes=1, init_embed_dim=96, window_size=7):
    inputs = layers.Input(shape=input_size)
    x = patch_partition(inputs, patch_size=4, embed_dim=init_embed_dim)

    skips, embed_dims = [], []
    embed_dim = init_embed_dim

    # Encoder
    for level in range(3):
        for _ in range(2):
            x = SwinTransformerBlock(
                num_heads=3 * (2 ** level),
                window_size=window_size
            )(x)
        skips.append(x)
        embed_dims.append(embed_dim)
        x = patch_merging(x, embed_dim)
        embed_dim *= 2

    # Bottleneck
    for _ in range(2):
        x = SwinTransformerBlock(
            num_heads=24,
            window_size=window_size
        )(x)

    # Decoder
    for level in reversed(range(3)):
        prev_embed_dim = embed_dim
        embed_dim = embed_dims[level]
        x = patch_expanding(x, prev_embed_dim)
        x = layers.Conv2D(embed_dim, kernel_size=1, padding='same')(x)
        skip = skips[level]
        skip = layers.Conv2D(embed_dim, kernel_size=1, padding='same')(skip)
        x = layers.Concatenate()([x, skip])
        for _ in range(2):
            x = SwinTransformerBlock(
                num_heads=3 * (2 ** level),
                window_size=window_size
            )(x)

    # Final upsampling + output
    x = patch_expanding(x, embed_dim * 2)
    x = patch_expanding(x, embed_dim)
    outputs = layers.Conv2D(num_classes, kernel_size=1, activation='sigmoid', dtype="float32")(x)

    return models.Model(inputs, outputs)