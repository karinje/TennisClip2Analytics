import torch

IMG_RESIZE = (360,640)


def mask2coord(batch, img_size=IMG_RESIZE):
    assert batch.ndim==3 or batch.ndim==4
    coords = []
    w = img_size[1]
    for element in batch:
        if element.ndim==3:
            element = element.argmax(axis=0)
        coords.append(torch.tensor([[element.argmax().item()%w, element.argmax().item()//w]]))
    return torch.cat(coords,dim=0).float()
        
if __name__=="__main__":
    inp_batch = torch.randint(low=0, high=255, size=(4,360,640))
    pred_batch = torch.randn((4,255,360,640))
    print(mask2coord(inp_batch).shape, mask2coord(pred_batch))
