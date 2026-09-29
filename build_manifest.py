import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset_root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--expect', type=int, default=0)
    a=p.parse_args()
    root=Path(a.dataset_root).resolve()
    img=root/'images'; labels=root/'labels_masks'
    if not img.is_dir() or not labels.is_dir():
        raise FileNotFoundError('Expected images/ and labels_masks/ directories')
    exts={'.jpg','.jpeg','.png','.webp','.bmp'}
    imgs=sorted(x for x in img.iterdir() if x.suffix.lower() in exts)
    found=[];stems=set()
    for file in imgs:
        label=labels/(file.stem+'.png')
        if not label.is_file():
            raise FileNotFoundError(f'Missing label for {file.name}: {label}')
        if file.stem in stems:
            raise ValueError(f'Duplicate image stem {file.stem}')
        stems.add(file.stem)
        found.append({'image_path':f'images/{file.name}', 'mask_path':f'labels_masks/{file.stem}.png',
                      'label':1.0, 'class_name':'unknown'})
    if not found:
        raise ValueError('No matched images/labels')
    if a.expect and len(found)!=a.expect:
        raise ValueError(f'Expected {a.expect} images, but found {len(found)}; check downloaded split')
    output=Path(a.output);output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w',encoding='utf-8') as f:
        for r in found:
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(f'Wrote {len(found)} candidate entries to {output}. VERIFY official split and label encodings before research use.')


if __name__=='__main__':
    main()
