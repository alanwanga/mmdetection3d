import array
from tqdm import tqdm
from pypcd import pypcd

def pcd2bin(pcd_path):
    pc = pypcd.PointCloud.from_fileobj(open(pcd_path)).pc_data
    points = []
    for i in range(pc.shape[0]):
        points.append(pc[i][0])
        points.append(pc[i][1])
        points.append(pc[i][2])
        points.append(pc[i][3])
        points.append(0)
    fileobj = open(pcd_path.replace("pcd", "bin"), mode='wb')
    outvalues = array.array('f')
    outvalues.fromlist(points)
    outvalues.tofile(fileobj)
    fileobj.flush()
    fileobj.close()

pcds = [l.strip() for l in open("/home/ssm-user/xiaorui/lidar/qualcomm/20220704_Qualcomm_package/pcd_list.txt", "r").readlines()]
for pcd in tqdm(pcds):
    pcd2bin(pcd)