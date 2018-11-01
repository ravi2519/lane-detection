# lane-detection

This reporitory is kind of an experiment with lane detection being taught in Udacity's Self Driving Car Nano Degree.
I have gone through the steps that ar required to do lane detection on an image or video.

For an image, run the bin/lane_detection.py using following command:
python -c "import lane_detection as ld; ld.create_image( '../resources/ROAD-1.jpg' )"

For a video, run the code using following command. The function call has two arguments - Input Video File and Output Video File.
python -c "import lane_detection as ld; ld.create_video( '../resources/challenege.mp4', '../resources/challenege_done.avi' )"
