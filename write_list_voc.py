from xml.dom.minidom import parse
from os import path
names_file = "./data/voc.names"
voc_root_dir = "./data/VOCdevkit"
anno_dir=path.join(voc_root_dir,"Annotations")
img_dir=path.join(voc_root_dir,"JPEGImages")

def parse_voc_xml(file_name, names_dict):
    result = []
    doc = parse(file_name)
    root = doc.documentElement
    size = root.getElementsByTagName('size')[0]
    width = int(size.getElementsByTagName('width')[0].childNodes[0].data)
    height = int(size.getElementsByTagName('height')[0].childNodes[0].data)
    result.append([str(width), str(height)])

    objs = root.getElementsByTagName('object')
    for obj in objs:
        name = obj.getElementsByTagName('name')[0].childNodes[0].data
        name_id = names_dict[name]

        bndbox = obj.getElementsByTagName('bndbox')[0]
        xmin = int(float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data))
        ymin = int(float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data))
        xmax = int(float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data))
        ymax = int(float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data))

        result.append([str(name_id), str(xmin), str(ymin), str(xmax), str(ymax)])
    return result


def word2id(names_file):
    id_dict = {}
    contents = []
    with open(names_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').strip()
            if len(line) == 0:
                continue
            contents.append(line)
    for i in range(len(contents)):
        id_dict[str(contents[i])] = i
    return id_dict


names_dict = word2id(names_file)
with open("data/my_data/train_test.txt", 'w') as f, open(path.join(voc_root_dir,"ImageSets/Main/train.txt"), 'r') as t:
    index = 0
    for file in t.readlines():
        file = file.strip('\n')+".jpg"
        f.write(str(index))
        f.write(" ")
        f.write(path.join(img_dir,file))
        f.write(" ")
        index += 1
        try:
            results = parse_voc_xml(path.join(anno_dir,file.replace("jpg", "xml")), names_dict)
            for result in results:
                for item in result:
                    f.write(item)
                    f.write(" ")
            f.write("\n")
        except:
            pass
