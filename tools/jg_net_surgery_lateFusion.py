#!/usr/bin/env python2

import caffe

PROTOTXT_INPUT1 = "/data/workspace/caffemodels/vgg16/nyud fcn 32s color test.prototxt"
#WEIGHTS_INPUT1 = "/data/workspace/caffemodels/vgg16/nyud-fcn32s-color-heavy.caffemodel"
WEIGHTS_INPUT1 = "/data/workspace/caffemodels/vgg16/nyud-fcn32s-hha-heavy.caffemodel"


PROTOTXT_INPUT2 = "/data/workspace/caffemodels/vgg16/nyud fcn 32s color test.prototxt"
#WEIGHTS_INPUT2 = "/data/workspace/caffemodels/vgg16/nyud-fcn32s-color-heavy.caffemodel"
WEIGHTS_INPUT2 = "/data/workspace/caffemodels/vgg16/nyud-fcn32s-hha-heavy.caffemodel"

PROTOTXT_OUTPUT = "/data/workspace/caffemodels/vgg16/sunrgbd37/deploy.prototxt"
WEIGHTS_OUTPUT = "/data/workspace/caffemodels/vgg16/sunrgbd37/fcn32s-heavy-nyud-hha-37.caffemodel"

def generate_fcn():

    caffe.set_mode_cpu()

    print "Loading output prototxt..."
    outputNet = caffe.Net(PROTOTXT_OUTPUT, caffe.TEST)

    #### INPUT 1
    print "Loading input proto+model 1..."
    inputNet = caffe.Net(PROTOTXT_INPUT1, WEIGHTS_INPUT1, caffe.TEST)
    print "Transplanting parameters of first net (prefix : _data0)..."
    transplant(outputNet, inputNet, '_data0')
    del inputNet

    #### INPUT 2
    print "Loading input proto+model 2..."
    inputNet = caffe.Net(PROTOTXT_INPUT2, WEIGHTS_INPUT2, caffe.TEST)
    print "Transplanting parameters of second net (prefix : _data1)..."
    transplant(outputNet, inputNet, '_data0')
    del inputNet

    #### SAVING
    print "Saving output model to %s " % WEIGHTS_OUTPUT
    outputNet.save(WEIGHTS_OUTPUT)


def transplant(new_net, net, suffix=''):
    # from fcn.berkeleyvision.org
    for p in net.params:
        print "Working on", p, ':'
        p_new = p + suffix
        if p_new not in new_net.params:
            print 'dropping', p
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p_new]) - 1):
                print 'dropping', p, i
                break
            if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                print 'coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape
            else:
                print 'copying', p, ' -> ', p_new, i
            new_net.params[p_new][i].data.flat = net.params[p][i].data.flat


if __name__ == '__main__':
    generate_fcn()
