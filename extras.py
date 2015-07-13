import code
import sys

def save_predictions(coord, dim, pred, gt, fname='pred.csv'):
    print 'Writing predictions to %s...' % fname
    with open(fname, 'w') as fout:
        fout.write('x,y,z,pred,ground_truth\n')
        for i in range(coord.shape[0]):
            fout.write('%d,%d,%d,%d,%d\n' % (coord[i,0], coord[i,1], coord[i,2], pred[i], gt[i]))
    with open(fname+'.dim', 'w') as fout2:
        fout2.write('%d,%d,%d\n' % (dim[0], dim[1], dim[2]))
    print 'Written.'

def keyboard(banner=None):
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print "# Use quit() to exit :) Happy debugging!"
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner, local=namespace)
    except SystemExit:
        return 

