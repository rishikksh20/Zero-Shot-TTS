from torch import nn
import torch
from module import Postnet
from module import PositionalEncoding
from transformer import FFTransformerBlock
import torch.nn.functional as F
from module import LengthRegulator, DurationPredictor, Conv1dBNReLU, make_pad_mask

class MelGenerator(nn.Module):
    def __init__(self, idim, hp):
        super().__init__()
        # use idx 0 as padding idx
        padding_idx = 0


        # Embedding
        self.text_embed = nn.Embedding(
                num_embeddings=idim, embedding_dim=512, padding_idx=padding_idx
            )

        self.text_encoder_prenet = nn.Sequential(
            Conv1dBNReLU(512, 512),
            nn.Dropout(hp.model.prenet_dropout),
            Conv1dBNReLU(512, 512),
            nn.Dropout(hp.model.prenet_dropout),
            Conv1dBNReLU(512, 512),
            nn.Dropout(hp.model.prenet_dropout),
            nn.Conv1d(512, hp.model.adim, 1)

        )


        #Encoder
        self.pos_enc = PositionalEncoding(hp.model.adim, dropout_p=0.1)
        self.encoder = FFTransformerBlock(hp.model.adim, hp.model.aheads, hp.model.eunits, hp.model.elayers,
                                          kernel_size_up=hp.model.positionwise_conv_kernel_size1,
                                          kernel_size_down=hp.model.positionwise_conv_kernel_size2, dropout_p=0.1)

        # Spectrogram Encoder
        self.input_layer = nn.Conv1d(hp.audio.num_mels, hp.model.adim, 1)
        self.spec_embed = nn.Conv1d(hp.model.adim, hp.model.adim, 1)
        self.spec_pos_enc = PositionalEncoding(hp.model.adim, dropout_p=0.1)
        self.spec_encoder = FFTransformerBlock(hp.model.adim, hp.model.sheads, hp.model.sunits, hp.model.slayers,
                                               kernel_size_up=hp.model.positionwise_conv_kernel_size1,
                                               kernel_size_down=hp.model.positionwise_conv_kernel_size2, dropout_p=0.1)


        # Duration Predictor
        self.duration_predictor = DurationPredictor(
            idim=hp.model.adim,
            n_layers=hp.model.duration_predictor_layers,
            n_chans=hp.model.duration_predictor_chans,
            kernel_size=hp.model.duration_predictor_kernel_size,
            dropout_rate=hp.model.duration_predictor_dropout_rate,
        )


        # Length regulator
        self.length_regulator = LengthRegulator()


        # Decoder
        self.pos_dec = PositionalEncoding(hp.model.ddim, dropout_p=0.1)
        self.decoder = FFTransformerBlock(hp.model.ddim, hp.model.aheads, hp.model.dunits, hp.model.dlayers,
                                          kernel_size_up=hp.model.positionwise_conv_kernel_size1,
                                          kernel_size_down=hp.model.positionwise_conv_kernel_size2, dropout_p=0.1)




        # Postnet
        self.postnet = (
            None
            if hp.model.postnet_layers == 0
            else Postnet(
                idim=hp.audio.num_mels,
                odim=hp.audio.num_mels,
                n_layers=hp.model.postnet_layers,
                n_chans=hp.model.postnet_chans,
                n_filts=hp.model.postnet_filts,
                use_batch_norm=hp.model.use_batch_norm,
                dropout_rate=hp.model.postnet_dropout_rate,
            )
        )


        self.spectrogram_out = nn.Linear(hp.model.adim, hp.audio.num_mels)

    def forward(self, text, duration, ilens, mel):
        '''
        inputs:
            text : [B, Lmax, Dim]
            duration: [B, Lmax]
        outputs :
            mel_spec : [B, Tmax, Bin]
        '''

        # Embedding
        emb = self.embed(text)                  # [B, Lmax, 512]
        emb = emb.transpose(1, -1).contiguous() # [B, 512, Lmax]

        emb = self.text_encoder_prenet(emb)     # [B, 256, Lmax]

        # Encoder
        emb = self.pos_enc(emb)                 # [B, 256, Lmax]
        hs, mask = self.encoder(emb, ilens)         # [B, 256, Lmax]

        # forward duration predictor and length regulator
        d_masks = make_pad_mask(ilens).to(text.device)

        d_outs = self.duration_predictor(hs.transpose(1, -1), d_masks)  # (B, Tmax)

        # Spectrogram Encoder
        spec_emb = F.relu(self.input_layer(mel.transpose(-2, -1)))  # [B, 256, Tmax]
        spec_emb = F.relu(self.spec_embed(spec_emb))
        spec_emb = self.spec_pos_enc(spec_emb)
        hs_spec, mask_spec = self.spec_encoder(spec_emb, ilens)  # [B, 256, Lmax]

        # Length Regulator
        hs = self.length_regulator(hs.transpose(1, -1), duration, ilens)    # [B, Tmax, 256]



        # Decoder
        hs = hs.transpose(1, -1).contiguous()       # [B, 256, Tmax]
        hs = hs + hs_spec
        olens = duration.sum(-1)  # [B]
        hs = self.pos_dec(hs)
        out, mask = self.decoder(hs, olens)             # [B, 256, Tmax]

        mel_spec = self.spectrogram_out(out.transpose(1, -1))       # [B, Tmax, 80]

        # Postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            mel_spec_fine = None
        else:
            mel_spec_fine = mel_spec + self.postnet(
                mel_spec.transpose(1, 2)
            ).transpose(1, 2)

        return mel_spec, mel_spec_fine, mask, olens, d_outs
