# Original Implementation from https://github.com/Nota-NetsPresso/BK-SDM/issues/36 by bokyeong1015:https://github.com/bokyeong1015


import torch
from torchvision import transforms as pth_transforms


class ImageEmbedder:
    def __init__(self, dino_model='dino_vits16', dino_n_last_blocks=4, dino_avgpool_patchtokens=False, device="cuda"):     
        self.device = device   
        # dino-related part
        self.dino_model = torch.hub.load('facebookresearch/dino:main', dino_model)
        self.dino_model.to(device)
        self.dino_model.eval()
        self.dino_preprocess = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.dino_n_last_blocks = dino_n_last_blocks
        self.dino_avgpool_patchtokens = dino_avgpool_patchtokens        
        # clip-related part
      

    def get_img_emb(self, pil_img, model_type):
        with torch.no_grad():
            if model_type == 'dino':
                image = self.dino_preprocess(pil_img).unsqueeze(0).to(self.device)
                intermediate_output = self.dino_model.get_intermediate_layers(image, self.dino_n_last_blocks)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if self.dino_avgpool_patchtokens:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)                
                image_feature = output            
            else:
                raise NotImplementedError
        # print(image_feature.shape)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)       
        return image_feature
    

