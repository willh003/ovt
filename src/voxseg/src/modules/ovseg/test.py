import os
if __name__ == "__main__":
    classes = ['traversable', 'untraversable', 'obstacle']
    inp_file = 'test_data/rgb_38.png'

    class_str = ''.join([f"'{elem}' " for elem in classes])

    cmd = f"python demo.py --config-file configs/ovseg_swinB_vitL_demo.yaml --class-names {class_str} --input {inp_file} --output ./pred --opts MODEL.WEIGHTS models/ovseg_swinbase_vitL14_ft_mpt.pth"
    print(cmd)
    os.system(cmd)