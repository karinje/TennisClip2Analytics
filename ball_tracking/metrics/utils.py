import torch

IMG_RESIZE = (360,640)


def mask2coord(batch, img_size=IMG_RESIZE):
      bs = len(batch)
      w = img_size[1]
      return torch.stack([torch.stack([x+1,y+1]) for x,y in zip(batch.view(bs,-1).argmax(dim=-1)%w, \
                                                                batch.view(bs,-1).argmax(dim=-1)//w)]
                                                                ).float()
