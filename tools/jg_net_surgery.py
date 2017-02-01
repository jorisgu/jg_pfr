#!/usr/bin/env python2

import caffe

PROTOTXT_INPUT = "/data/workspace/caffemodels/vgg16/nyud fcn 32s color test.prototxt"
#WEIGHTS_INPUT = "/data/workspace/caffemodels/vgg16/nyud-fcn32s-color-heavy.caffemodel"

WEIGHTS_INPUT = "/data/workspace/caffemodels/vgg16/nyud-fcn32s-hha-heavy.caffemodel"

PROTOTXT_OUTPUT = "/data/workspace/caffemodels/vgg16/sunrgbd37/deploy.prototxt"
WEIGHTS_OUTPUT = "/data/workspace/caffemodels/vgg16/sunrgbd37/fcn32s-heavy-nyud-hha-37.caffemodel"

def generate_fcn():

    caffe.set_mode_cpu()

    print "Loading input proto+model..."
    inputNet = caffe.Net(PROTOTXT_INPUT, WEIGHTS_INPUT, caffe.TEST)

    print "Loading output prototxt..."
    outputNet = caffe.Net(PROTOTXT_OUTPUT, caffe.TEST)

    print "Transplanting parameters..."
    transplant(outputNet, inputNet)

    new_layer='upscore37'
    layer='upscore'
    dim=37
    print "Keeping bilinear weights..."
    bilin(outputNet,new_layer, inputNet,layer,dim)

    print "Saving output model to %s " % WEIGHTS_OUTPUT
    outputNet.save(WEIGHTS_OUTPUT)

def bilin(new_net,new_layer, net, layer, dim):
    new_net.params[new_layer][0].data[...]=net.params[layer][0].data[0:dim,0:dim]



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
