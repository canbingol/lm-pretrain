
import sentencepiece as spm
import torch

@torch.no_grad()
def generate(model, sp, prompt, device="cuda", max_new_tokens=64):
    model.to(device).eval()
    ids = sp.encode(prompt, out_type=int)
    
    bos = sp.bos_id() if hasattr(sp, "bos_id") else -1
    if bos is not None and bos >= 0 and (not ids or ids[0] != bos):
        ids = [bos] + ids

    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]  # [1,T]
    eos = sp.eos_id() if hasattr(sp, "eos_id") else -1

    for _ in range(max_new_tokens):
        logits = model(x)[:, -1, :]          # [1,V]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)  # greedy [1,1]
        x = torch.cat([x, nxt], dim=1)
        if eos is not None and eos >= 0 and int(nxt.item()) == eos:
            break

    return sp.decode(x[0].tolist())