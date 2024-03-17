import clip
import torch
from PIL import Image
import numpy as np
import PIL
import torch.nn.functional as F
import os
from tqdm import tqdm
from PIL import Image
from sklearn import metrics
from dino_distance import ImageEmbedder


device = 'cuda'
model, preprocess_clip = clip.load("ViT-B/32", device=device)
dino_model = None


def calculate_auc_asr_stat(nonmember_scores, member_scores):

    total = member_scores.size(0) + nonmember_scores.size(0)

    min_score = min(member_scores.min(), nonmember_scores.min()).item()
    max_score = max(member_scores.max(), nonmember_scores.max()).item()
    TPR_list = []
    FPR_list = []

    best_asr = 0

    TPRatFPR_1 = 0
    FPR_1_idx = 999
    TPRatFPR_01 = 0
    FPR_01_idx = 999

    for threshold in torch.range(min_score, max_score, (max_score - min_score) / 1000):
        acc = ((member_scores >= threshold).sum() + (nonmember_scores < threshold).sum()) / total

        TP = (member_scores >= threshold).sum()
        TN = (nonmember_scores < threshold).sum()
        FP = (nonmember_scores >= threshold).sum()
        FN = (member_scores < threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        ASR = (TP + TN) / (TP + TN + FP + FN)

        if ASR > best_asr:
            best_asr = ASR

        if FPR_1_idx > (0.01 - FPR).abs():
            FPR_1_idx = (0.01 - FPR).abs()
            TPRatFPR_1 = TPR

        if FPR_01_idx > (0.001 - FPR).abs():
            FPR_01_idx = (0.001 - FPR).abs()
            TPRatFPR_01 = TPR

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())

        # print(f'Score threshold = {threshold:.16f} \t ASR: {acc:.4f} \t TPR: {TPR:.4f} \t FPR: {FPR:.4f}')
    auc = metrics.auc(np.asarray(FPR_list), np.asarray(TPR_list))
    return auc
    # print(f'AUC: {auc} \t ASR: {best_asr} \t TPR@FPR=1%: {TPRatFPR_1} \t TPR@FPR=0.1%: {TPRatFPR_01}')


def clip_feature(input_image, model_type='clip'):
    if model_type == 'clip':
        image = preprocess_clip(Image.open(input_image)).unsqueeze(0).to(device)
        image_feature = model.encode_image(image)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)      
    elif model_type == 'dino':
        image_feature = dino_model.get_img_emb(Image.open(input_image), model_type)
    return image_feature


def clip_sc(path1, path2):
    feature1 = clip_feature(path1)
    feature2 = clip_feature(path2)
    return max(F.cosine_similarity(feature1, feature2).item(), 0)


@ torch.no_grad()
def clip_dir_cos_sim(src_dir, gen_dir, model_type='clip'):
    gen_imgs = os.listdir(gen_dir)
    src_imgs = os.listdir(src_dir)
    if model_type == 'clip':
        dimension_num = 512
    elif model_type == 'dino':
        dimension_num = 1536
    src_embedding = torch.zeros([len(src_imgs), dimension_num])
    gen_embedding = torch.zeros([len(gen_imgs), dimension_num])
    for i in tqdm(range(len(src_imgs))):
        img_id = src_imgs[i]
        img_name = os.path.join(src_dir, img_id)
        src_embedding[i] = clip_feature(img_name, model_type=model_type)[0]
    for i in tqdm(range(len(gen_imgs))):
        if gen_imgs[i] == '.ipynb_checkpoints':
            continue
        if gen_imgs[i][-11:-8] == 'src' or gen_imgs[i][0:3] == 'low':
            continue

        img_id = gen_imgs[i]
        img_name = os.path.join(gen_dir, img_id)
        gen_embedding[i] = clip_feature(img_name, model_type=model_type)[0]
    cosine_sim = F.cosine_similarity(src_embedding[:, None, :], gen_embedding[None, :, :], dim=2)
    max_similarity, max_indices = torch.max(cosine_sim, dim=1)
    return max_similarity, max_indices



def find_threshold(member_list, non_member_list):
    min_value = torch.min(torch.min(member_list), torch.min(non_member_list))
    max_value = torch.max(torch.max(member_list), torch.max(non_member_list))
    corr_count = 0
    optimal_threshold = 0
    for i in range(1000):
        threshold = min_value + (max_value-min_value)*(i+1)/1000
        corr_num = torch.sum(member_list<threshold)+torch.sum(non_member_list>=threshold)
        if corr_num>corr_count:
            corr_count = corr_num
    return corr_count/(member_list.shape[0] + non_member_list.shape[0])


def main(training_mode = "db_prior",
        num_imgs = 10,
        version = "1.4",
        per_step = 50,
        src_dataset_mode = 0,
         model_type = 'clip',
         img_num = 1000,
         atk_type = "True_216_70.0False_2.0_1000"):
    

    assert version == "1.4"
    
    if src_dataset_mode == 0:
        src_dataset_path = "wikiart_vangogh"
    elif src_dataset_mode == 1:
        src_dataset_path = "object_dog"

    data_src = f"{src_dataset_path}/{num_imgs}_{per_step}_v{version}/"
    src_style_dirs = f"{training_mode}/{data_src}"
    type_list = ['membership', 'hold_out']
    result_list = [[], []]
    global dino_model
    if model_type == 'dino':
        dino_model = ImageEmbedder()
    
    for type_name in type_list:
        for style_name in os.listdir(f"./datasets/{src_dataset_path}/{num_imgs}"):
            src_imgs_dir = f"./datasets/{src_dataset_path}/{num_imgs}/{style_name}/{type_name}"
            extracted_dir = f"Recovered_Samples/{src_style_dirs}/{atk_type}/{style_name}/{type_name}"

            sim_list = clip_dir_cos_sim(src_imgs_dir, extracted_dir, model_type)[0]
            for sim_value in sim_list:
                result_list[type_list.index(type_name)].append(sim_value)

    acc, auc = find_threshold(torch.Tensor(result_list[1]), torch.Tensor(result_list[0])), calculate_auc_asr_stat(torch.Tensor(result_list[1]), torch.Tensor(result_list[0]))
    return acc, auc

if __name__ == "__main__":
    acc, auc = main(model_type = 'clip', src_dataset_mode = 0, num_imgs = 10)
    print(f'Current Acc. is {acc}, AUC is {auc}')
    acc, auc = main(model_type = 'dino', src_dataset_mode = 0, num_imgs = 10)
    print(f'Current Acc. is {acc}, AUC is {auc}')
    acc, auc = main(model_type = 'clip', src_dataset_mode = 1, num_imgs = 2)
    print(f'Current Acc. is {acc}, AUC is {auc}')
    acc, auc = main(model_type = 'dino', src_dataset_mode = 1, num_imgs = 2)
    print(f'Current Acc. is {acc}, AUC is {auc}')


