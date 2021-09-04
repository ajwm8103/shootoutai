SET /P arg1=Enter a framerate: 
SET /P arg2=Enter a start_number: 
SET /P arg3=Enter a filename (.mp4 automatic): 

cd "C:\Users\whmra\OneDrive\Documents\Python Projcs\STABLEBASELINES\1v1\tools\cool moments\torender"
ffmpeg -r %arg1% -start_number %arg2% -i frame_%%03d.jpeg -c:v libx264 -vf fps=%arg1% -pix_fmt yuv420p ../files/%arg3%.mp4