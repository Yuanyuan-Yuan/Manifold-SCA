import os
import sys

cpu_id = int(sys.argv[1])
seg_id = int(sys.argv[2])
os.system('taskset -c %d python pp_audio.py %d %d' % (cpu_id, cpu_id, seg_id))