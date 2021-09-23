import torch
import torch.nn as nn
import torch.nn.functional as F

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.cpu().unsqueeze(1).expand(-1, max_len)

    return mask

class FFTransformer(nn.Module):
    def __init__(self, in_out_channels, num_heads, hidden_channels_ffn=1024, kernel_size_up=9, kernel_size_down=1,
                 dropout_p=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(in_out_channels, num_heads, dropout=dropout_p)

        self.conv1 = nn.Conv1d(in_out_channels, hidden_channels_ffn, kernel_size=kernel_size_up,
                               padding=(kernel_size_up - 1) // 2)
        self.conv2 = nn.Conv1d(hidden_channels_ffn, in_out_channels, kernel_size=kernel_size_down,
                               padding=(kernel_size_down - 1) // 2)

        self.norm1 = nn.LayerNorm(in_out_channels)
        self.norm2 = nn.LayerNorm(in_out_channels)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """😦 ugly looking with all the transposing"""
        src = src.permute(2, 0, 1).contiguous()
        src2, enc_align = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + src2)
        # T x B x D -> B x D x T
        src = src.permute(1, 2, 0).contiguous()
        src2 = self.conv2(F.relu(self.conv1(src)))
        src2 = self.dropout(src2)
        src = src + src2
        src = src.transpose(1, 2).contiguous()
        src = self.norm2(src)
        src = src.transpose(1, 2).contiguous()

        return src, enc_align


class FFTransformerBlock(nn.Module):
    def __init__(self, in_out_channels, num_heads, hidden_channels_ffn, num_layers, kernel_size_up=9, kernel_size_down=1,
                 dropout_p=0.1):
        super().__init__()
        self.fft_layers = nn.ModuleList(
            [
                FFTransformer(
                    in_out_channels=in_out_channels,
                    num_heads=num_heads,
                    hidden_channels_ffn=hidden_channels_ffn,
                    kernel_size_up=kernel_size_up,
                    kernel_size_down=kernel_size_down,
                    dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, length=None):  # pylint: disable=unused-argument
        """
        TODO: handle multi-speaker
        Shapes:
            x: [B, C, T]
            mask:  [B, 1, T] or [B, T]
        """
        mask = get_mask_from_lengths(length.cuda())
        alignments = []
        for layer in self.fft_layers:
            x, align = layer(x, src_key_padding_mask=mask.cuda())

            alignments.append(align.unsqueeze(1))
        alignments = torch.cat(alignments, 1)

        return x, mask