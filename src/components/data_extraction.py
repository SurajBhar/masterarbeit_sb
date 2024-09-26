import os
import cv2
import csv

class FrameExtractor:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir

    def extract_frames(self, row, max_frames_per_chunk):
        participant_id, file_id, annotation_id, frame_start, frame_end, activity, chunk_id = row
        video_filepath = os.path.join(self.data_dir, file_id + '.mp4')
        new_file_id = file_id.replace("/", "_")
        updated_output_dir = os.path.join(self.output_dir, f'{activity}', f'{participant_id}_{new_file_id}_frames_{frame_start}_{frame_end}_ann_{annotation_id}_chunk_{chunk_id}')
        
        os.makedirs(updated_output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_filepath)
        frame_count = 0
         
        try:
            # Set the starting frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_start))
            
            for frame_num in range(int(frame_start), int(frame_end) + 1):
                ret, frame = cap.read()
                if not ret:
                    print(f"Frame number {frame_num} is missing.")
                    break
                #new_file_id = file_id.replace("/", "_")
                output_filename = f'img_{frame_num:06d}.png'
                output_path = os.path.join(updated_output_dir, output_filename)
                cv2.imwrite(output_path, frame)
                frame_count += 1
                
                if frame_count > max_frames_per_chunk:
                    break
        finally:
            cap.release()

def process_annotations(annotation_file, data_dir, root_dataset_dir, dataset_sub_dir, max_frames_per_chunk=48):
    with open(annotation_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        output_dir = os.path.join(root_dataset_dir, dataset_sub_dir)
        frame_extractor = FrameExtractor(data_dir, output_dir)
        
        for row in reader:
            frame_extractor.extract_frames(row, max_frames_per_chunk)

# First : Train : Split_1, 
def main():
    data_dir = "/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline/data/kinect_color/kinect_color"
    # "/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline/daa_dataset"
    root_dataset_dir = "/net/polaris/storage/deeplearning/sur_data/rgb_daa/split_1"
    dataset_sub_dirs = ['train'] #, 'train', 'test', 'val'
    annotation_files = [
        '/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline/data/kinect_color_annotation/activities_3s/kinect_color/midlevel.chunks_90.split_1.train.csv'
    ] #,
      #  '/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline/data/kinect_color_annotation/activities_3s/kinect_color/midlevel.chunks_90.split_0.train.csv',
      #  '/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline/data/kinect_color_annotation/activities_3s/kinect_color/midlevel.chunks_90.split_0.test.csv'
    
    for annotation_file, dataset_sub_dir in zip(annotation_files, dataset_sub_dirs):
        process_annotations(annotation_file, data_dir, root_dataset_dir, dataset_sub_dir)

if __name__ == "__main__":
    main()
