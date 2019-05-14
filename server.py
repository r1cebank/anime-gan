# import libraries
print('importing libraries...')
from flask import Flask, request, jsonify
from flask import send_file
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from io import BytesIO
from model import Generator


PATH = "model.pth.tar"

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64
ngpu = 0


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
netG = Generator(ngpu, nz, ngf, nc).to(device)
netG.load_state_dict(torch.load(PATH, map_location='cpu'))

# set flask params
app = Flask(__name__)
@app.route("/")
def root():
    return "Image generation example\n"

@app.route("/generate")
def generate():
    noise = torch.randn(1, nz, 1, 1, device=device)
    fake = netG(noise).detach().cpu()
    fake = fake.permute(1, 2, 3, 0)
    fake = fake.squeeze()
    z = fake * torch.tensor(std).view(3, 1, 1)
    z = z + torch.tensor(mean).view(3, 1, 1)
    img_io = BytesIO()
    pil_img = transforms.ToPILImage()(z)
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=3939)
