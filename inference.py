import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn as nn
from models.TMC import ETMC
from torchsummary import summary
from models import image

# Set random seed for reproducibility.
torch.manual_seed(42)


# Define the audio_args dictionary
audio_args = {
    'nb_samp': 64600,
    'first_conv': 1024,
    'in_channels': 1,
    'filts': [20, [20, 20], [20, 128], [128, 128]],
    'blocks': [2, 4],
    'nb_fc_node': 1024,
    'gru_node': 1024,
    'nb_gru_layer': 3,
}


def get_args(parser):
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default="datasets/train/fakeavceleb*")
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=1024)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="MMDF")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--savedir", type=str, default="./savepath/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--annealing_epoch", type=int, default=10)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--pretrained_image_encoder", type=bool, default = False)
    parser.add_argument("--freeze_image_encoder", type=bool, default = False)
    parser.add_argument("--pretrained_audio_encoder", type = bool, default=False)
    parser.add_argument("--freeze_audio_encoder", type = bool, default = False)
    parser.add_argument("--augment_dataset", type = bool, default = True)

    for key, value in audio_args.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)

def model_summary(args):
    '''Prints the model summary.'''
    model = ETMC(args)
    summary(model, input_size=[(1, args.nb_samp), (3, args.FINE_SIZE, args.FINE_SIZE)])

def load_multimodal_model(args):
    '''Load multimodal model'''
    model = ETMC(args)
    ckpt = torch.load('checkpoints\model_best.pt', map_location = torch.device('cpu'))
    model.load_state_dict(ckpt,strict = False)
    model.eval()
    return model

def load_img_modality_model(args):
    '''Loads image modality model.'''
    rgb_encoder = image.ImageEncoder(args)
    ckpt = torch.load('checkpoints\model_best.pt', map_location = torch.device('cpu'))
    rgb_encoder.load_state_dict(ckpt,strict = False)
    rgb_encoder.eval()
    return rgb_encoder

def load_spec_modality_model(args):
    spec_encoder = image.RawNet(args)
    ckpt = torch.load('checkpoints\model_best.pt', map_location = torch.device('cpu'))
    spec_encoder.load_state_dict(ckpt,strict = False)
    spec_encoder.eval()
    return spec_encoder

def preprocess_img(face):
    face = face / 255
    face = cv2.resize(face, (256, 256))
    face = face.permute(2, 0, 1) #(W, H, C) -> (C, W, H)
    face = torch.unsqueeze(face, dim = 0) 
    face_pt = torch.Tensor(face)
    return face_pt

def preprocess_audio(audio_file):
    audio = torch.unsqueeze(audio_file, dim = 0)
    audio_pt = torch.Tensor(audio)
    return audio_pt

def deepfakes_spec_predict(input_audio, spec_model, multimodal):
    audio = preprocess_audio(input_audio)

    spec_grads = spec_model(audio)
    multimodal_grads = multimodal.spec_depth[0](spec_grads)

    out = nn.Softmax(dim=-1)(multimodal_grads)
    max_value, max_index = torch.max(out, dim=-1)

    preds = round(float(max_value) * 100, 3)
    if max_value > 0.5:
        text2 = f"The audio is REAL. \n Deepfakes Confidence: {preds}%"
    else:
        text2 = f"The audio is FAKE. \n Deepfakes Confidence: {preds}%"

    return max_index, max_value, text2

def deepfakes_image_predict(input_image, img_model, multimodal):
    face = preprocess_img(input_image)

    img_grads = img_model(face)
    multimodal_grads = multimodal.clf_rgb[0](img_grads)

    out = nn.Softmax(dim=-1)(multimodal_grads)
    max_value, max_index = torch.max(out, dim=-1)

    preds = round(float(max_value) * 100, 3)
    if max_value > 0.5:
        text2 = f"The image is REAL. \n Deepfakes Confidence: {preds}%"
    else:
        text2 = f"The image is FAKE. \n Deepfakes Confidence: {preds}%"

    return max_index, max_value, text2


def preprocess_video(input_video, n_frames = 5):
    v_cap = cv2.VideoCapture(input_video)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pick 'n_frames' evenly spaced frames to sample
    if n_frames is None:
        sample = np.arange(0, v_len)
    else:
        sample = np.linspace(0, v_len - 1, n_frames).astype(int)

    #Loop through frames.
    frames = []
    for j in range(v_len):
        success = v_cap.grab()
        if j in sample:
            # Load frame
            success, frame = v_cap.retrieve()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = preprocess_img(frame)
            frames.append(frame)
    v_cap.release()
    return frames


def deepfakes_video_predict(input_video, img_model, multimodal):
    video_frames = preprocess_video(input_video)
    real_grads = []
    fake_grads = []

    for face in video_frames:
        img_grads = img_model(face)
        multimodal_grads = multimodal.clf_rgb[0](img_grads)

        out = nn.Softmax(dim=-1)(multimodal_grads)
        real_grads.append(float(out[0]))
        fake_grads.append(float(out[1]))

    real_grads_mean = np.mean(real_grads)
    fake_grads_mean = np.mean(fake_grads)

    res = round(real_grads_mean * 100, 3)
    if real_grads_mean > fake_grads_mean:
        text = f"The video is REAL. \n Deepfakes Confidence: {res}%"
    else:
        text = f"The video is FAKE. \n Deepfakes Confidence: {res}%"

    return text

def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert not remaining_args, remaining_args
    model_summary(args)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()