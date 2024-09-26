import os
import sys
import getpass

def setup_ccname():
    user=getpass.getuser()
    # check if k5start is running, exit otherwise
    try:
        pid=open("/tmp/k5pid_"+user).read().strip()
        os.kill(int(pid), 0)
    except:
        sys.stderr.write("Unable to setup KRB5CCNAME!\nk5start not running!\n")
        sys.exit(1)
    try:
        ccname=open("/tmp/kccache_"+user).read().split("=")[1].strip()
        os.environ['KRB5CCNAME']=ccname
    except:
        sys.stderr.write("Unable to setup KRB5CCNAME!\nmaybe k5start not running?\n")
        sys.exit(1)

