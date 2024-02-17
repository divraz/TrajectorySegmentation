import os
import cv2
import json
import math
import shutil
import pickle
import hashlib
import jsonlines
import numpy as np
from tqdm import tqdm
from rlbench.const import DATA_PATH
from multiprocessing import Process

data_path = f"{DATA_PATH}data_random"
dataset_path = f"{DATA_PATH}dataset_random"
json_path = f"{dataset_path}/json"
video_path = f"{dataset_path}/videos"
action_path = f"{dataset_path}/actions"
gripper_path = f"{dataset_path}/gripper"

fps = 10
res = (224, 224)

def with_opencv(filename):
    video = cv2.VideoCapture(filename)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = int (frame_count / fps)
    return duration, frame_count

def save_video(path, hash_id, name):
    img_path = [name for name in os.listdir(path)]
    duration = len (img_path) / fps
    img_path = sorted(img_path, key = lambda x: int(x.split('.')[0]))
    img_path = img_path + [img_path[-1]] * (2*fps)
    images = [cv2.imread(f"{path}/{x}") for x in img_path]
    x = [a for a in images if a is None]
    if len (x) > 0:
        print (path)
    #return duration
    height, width, channels = images[0].shape

    os.makedirs(os.path.dirname(video_path + '/'), exist_ok=True)
    os.makedirs(os.path.dirname(video_path + '/' + hash_id + '/'), exist_ok=True)

    video = cv2.VideoWriter(
            video_path + '/' + hash_id + '/' + name + '.mp4',
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
            (height, width))

    for image in images:
        video.write (image)
        #video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    video.release()
    return int(duration)

def save_action(hash_id, action):
    action += [[0.0] * 21] * (2*fps)
    os.makedirs(os.path.dirname(action_path + '/'), exist_ok=True)
    np.savez (f"{action_path}/{hash_id}", features = np.array(action, dtype='float32')[::fps])

def save_gripper(hash_id, gripper):
    gripper += [[0.0] * 16] * (2*fps)
    os.makedirs(os.path.dirname(gripper_path + '/'), exist_ok=True)
    np.savez (f"{gripper_path}/{hash_id}", features = np.array(gripper, dtype='float32')[::fps])

def std_dev (test_list):
    mean = sum(test_list) / len(test_list) 
    variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list) 
    res = variance ** 0.5
    return res

def map_to_csv (i, task):
    print (i, task)
    duration_stats = []
    skill_stats = []
    writer_map = jsonlines.open (f"{dataset_path}/{task}_map.jsonl", mode = "w")

    variations = [name for name in os.listdir (f"{data_path}/{task}")]
    for variation in variations:
        episodes = [name for name in os.listdir (f"{data_path}/{task}/{variation}/episodes/")]
        for episode in episodes:
            temp_path = f"{data_path}/{task}/{variation}/episodes/{episode}/"
            
            # maintain a map for reference
            temp = {'data_path' : temp_path, "auto_id": hashlib.md5 (temp_path.encode ()).hexdigest ()}
            writer_map.write (temp)
            writer = jsonlines.open (f"{json_path}/{task}/{temp['auto_id']}.jsonl", mode = "w")
            #print (temp)
            
            # copy images to centralized video folder with md5 after converting to video
            duration = save_video (f"{temp_path}/front_rgb", temp['auto_id'], 'front')
            duration_stats.append (duration)
            save_video (f"{temp_path}/overhead_rgb", temp['auto_id'], "overhead")
            save_video (f"{temp_path}/wrist_rgb", temp['auto_id'], "wrist")
                
            frame_count = fps

            # read the pickle file and convert to actual data
            with open (f"{temp_path}low_dim_obs.pkl", "rb") as openfile:
                demo = pickle.load (openfile)

            instructions = demo.instructions
            change_point = {}
            for i, x in enumerate (demo.change_point):
                if x not in change_point.keys ():
                    change_point[x] = [9999, 0]
                change_point[x][0] = min (change_point[x][0], i)
                change_point[x][1] = max (change_point[x][1], i)
                
            for i, value in change_point.items ():
                value[0] = math.floor(value[0] / fps)
                value[1] = math.ceil(value[1] / fps)
                skill_stats.append (value[1] - value[0])
                #value[0] = value[0] / fps
                #value[1] = value[1] / fps

                #if value[1] < duration:
                #    value[1] += 2

            actions = []
            gripper = []

            for i in range (len(demo)):
                actions.append ( #21
                    demo[i].joint_forces.tolist () #7
                    + demo[i].joint_velocities.tolist () #7
                    + demo[i].joint_positions.tolist () #7
                )
                gripper.append ( #16
                    demo[i].gripper_joint_positions.tolist () #2
                    + [demo[i].gripper_open] #1
                    + demo[i].gripper_pose.tolist () #7
                    + demo[i].gripper_touch_forces.tolist () #7
                )
            #print (len(gripper))
            #for i, x in enumerate(gripper[-1]):
            #    print (i, x, type(x))   
            #print (gripper)
            save_action (temp['auto_id'], actions)
            save_gripper (temp['auto_id'], gripper)
            for instruction_set in instructions:
                for i, instruction in enumerate(instruction_set):
                    #if i >= len(change_point):
                    #    continue
                    if 'SKILL_' in instruction:
                        continue
                    try:
                        query = {
                                "query": instruction,
                                "duration": duration + 2,
                                "vid": temp['auto_id'],
                                "relevant_windows": [change_point[i]],
                                #"relevant_clip_ids": [int(i/2) for i in range ((int(change_point[i][0])//2)*2, int(change_point[i][1]), 2)]
                                "relevant_clip_ids": [int(i) for i in range (int(change_point[i][0]), int(change_point[i][1]) + 1)]
                        }
                        query['qid'] = hashlib.md5 (query['query'].encode ()).hexdigest ()
                        query['saliency_scores'] = [[4, 4, 4] for i in range (len (query['relevant_clip_ids']))]
                        writer.write (query)
                    except Exception as e:
                        print ('exception', temp_path, e)
            writer.close ()
    writer_map.close ()
    print (task, 'duration', len(duration_stats), min(duration_stats), max(duration_stats), sum(duration_stats)/len(duration_stats), std_dev(duration_stats))
    print (task, 'skills', len(skill_stats), min(skill_stats), max(skill_stats), sum(skill_stats)/len(skill_stats), std_dev(skill_stats))

if __name__ == '__main__':
    
    os.makedirs(os.path.dirname(json_path + '/'), exist_ok=True)
    
    tasks = [name for name in os.listdir (f"{data_path}")]
    for task in tasks:
        if not os.path.exists (f"{json_path}/{task}"):
            os.makedirs(f"{json_path}/{task}")
    
    processes = [Process(
        target=map_to_csv, args=(i, tasks[i])) 
        for i in range (len(tasks)
    )]
    
    [t.start() for t in processes]
    [t.join() for t in processes]
