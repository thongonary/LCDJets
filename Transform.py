import glob
import h5py
import pandas as pd
import numpy as np
from scipy import misc
import time
import sys
import matplotlib
from skimage import draw 
import os
import matplotlib.pyplot as plt
import random

features = ['Energy', 'Px', 'Py', 'Pz', 'Pt', 'Eta', 'Phi', 
                    'vtxX', 'vtxY', 'vtxZ','ChPFIso', 'GammaPFIso', 'NeuPFIso',
                    'isChHad', 'isNeuHad', 'isGamma', 'isEle',  'isMu', 
                        #'Charge'
           ]


# In[259]:

colors = {'isMu' : 'green',
        'isEle': 'green',
         'isGamma':'blue',
         'isChHad' : 'red',
         'isNeuHad': 'yellow'}

shapes = {'isMu' : 5,
          'isEle': 5,
          'isGamma':3,
          'isChHad' : 4,
          'isNeuHad': 0}

c_colors = [colors[k] for k in features[13:]]+['black']
cc_colors = [np.asarray(matplotlib.colors.to_rgb(k)) for k in c_colors]
cc_shapes = [shapes[k] for k in features[13:]]+[0]

def showSEvent(d,i,show=True):
    data = d[int(i),...]
    max_eta = 5
    max_phi = np.pi
    res= 100
    neta = int(max_eta*res)
    nphi = int(max_phi*res)
    eeta = 2.*max_eta / float(neta)
    ephi = 2.*max_phi / float(nphi)
    def ieta( eta ): return (eta+max_eta) / eeta
    def iphi(phi) : return (phi+max_phi) / ephi
    blend = 0.3
    image = np.ones((neta,nphi,3), dtype = np.float32)
    for ip in range(data.shape[0]):
        p_data = data[ip,:]
        eta = p_data[0]
        phi = p_data[1]
        if eta==0 and phi==0: 
            #print ip
            continue
        #pT = p_data[2]
        #lpT = min(max(np.log(pT)/5.,0.001), 10)*res/2.
        lpT = p_data[2]
        ptype = int(p_data[3])
        c = cc_colors[ ptype ]
        s = cc_shapes[ ptype ]
        R = lpT * res/1.5
        iee = ieta(eta)
        ip0 = iphi(phi)
        ip1 = iphi(phi+2*np.pi)
        ip2 = iphi(phi-2*np.pi)
        
        if s==0:
            xi0,yi0 = draw.circle(  iee, ip0,radius=R, shape=image.shape[:2])
            xi1,yi1 = draw.circle( iee, ip1, radius=R, shape=image.shape[:2])
            xi2,yi2 = draw.circle( iee, ip2, radius=R, shape=image.shape[:2]) 
            #if ptype == 5:
            #    print "MET",eta,phi
        else:
            nv = s
            vx = [iee + R*np.cos(ang) for ang in np.arange(0,2*np.pi, 2*np.pi/nv)]
            vy = [ip0 + R*np.sin(ang) for ang in np.arange(0,2*np.pi, 2*np.pi/nv)]
            vy1 = [ip1 + R*np.sin(ang) for ang in np.arange(0,2*np.pi, 2*np.pi/nv)]
            vy2 = [ip2 + R*np.sin(ang) for ang in np.arange(0,2*np.pi, 2*np.pi/nv)]
            xi0,yi0 = draw.polygon( vx, vy , shape=image.shape[:2])
            xi1,yi1 = draw.polygon( vx, vy1 , shape=image.shape[:2])
            xi2,yi2 = draw.polygon( vx, vy2 , shape=image.shape[:2])
            
        xi = np.concatenate((xi0,xi1,xi2))
        yi = np.concatenate((yi0,yi1,yi2))
        image[xi,yi,:] = (image[xi,yi,:] *(1-blend)) + (c*blend)
    

    if show:
        fig = plt.figure( frameon=False)
        plt.imshow(image.swapaxes(0,1))
        plt.axis('off')
        plt.savefig('fig.png', dpi=100,bbox_inches='tight', pad_inches=0)
        plt.show()
    return image
        


# In[316]:

def do_it_all( sample ,limit=None ):
    start = time.mktime(time.gmtime())
    dataset = None
    N=100
    max_I = limit if limit else sample.shape[0]
    for i in range(max_I):
        if i%N==0: 
            now = time.mktime(time.gmtime())
            so_far = now-start
            print i, so_far,"[s]"
            if i:
                eta = (so_far/i* max_I) - so_far
                print "finishing in", int(eta),"[s]", int(eta/60.),"[m]"
        img = showSEvent(sample, i, show=False)
        if dataset is None:
            dataset = np.zeros((max_I,)+img.shape)
            print dataset.shape
        dataset[i,...] = img
    return dataset


# In[325]:

def nf( fn ):
    return     fn.rsplit('/',1)[0]+'/images/'+fn.rsplit('/',1)[-1]

def move_to_thong(fn):
    if "train" in fn:
        return "/bigdata/shared/WMA/LCDJets/train/"+fn.rsplit('/',1)[-1]
    if "val" in fn:
        return "/bigdata/shared/WMA/LCDJets/val/"+fn.rsplit('/',1)[-1]

def make_reduced( f ) :
    if type(f) == str:
        f = h5py.File(f)    
    pf = f['Particles']
    reduced = np.zeros( (pf.shape[0], 801, 4))
    reduced[...,0] = f['Particles'][...,features.index('Eta')] 
    reduced[...,1] = f['Particles'][...,features.index('Phi')] 
    #reduced[...,2] = f['Particles'][...,features.index('Pt')] 
    reduced[...,2] = np.minimum(np.log(np.maximum(f['Particles'][...,features.index('Pt')], 1.001))/5., 10)
    reduced[...,3] = np.argmax( f['Particles'][..., 13:], axis=-1)

    h_reduced = np.zeros( (pf.shape[0], 1, 4))
    #h_reduced[...,0,2] = f['HLF'][..., 1] # MET
    h_reduced[...,0,2] = np.minimum(np.maximum(np.log(f['HLF'][..., 1])/5.,0.001), 10) # MET
    h_reduced[...,0,1] = f['HLF'][..., 2] # MET-phi
    h_reduced[...,0,3 ] = int(5) ## met type

    reduced = np.concatenate( (reduced, h_reduced), axis=1)

    return reduced

def convert_sample( fn, limit=None ):
    f = h5py.File(fn)    
    reduced = make_reduced(f)
    #new_fn = nf(fn)
    new_fn = move_to_thong(fn)
    print "Converting",fn,"into",new_fn,("for %s events"%limit) if limit else ""
    ds = do_it_all( reduced ,limit)
    n_f = h5py.File( new_fn,'w')
    #n_f['data'] = reduced 
    #n_f['Images'] = ds
    #n_f['Labels'] = f['Labels'][:limit,...] if limit else f['Labels'][...]
    if not np.isnan(ds).any():
        tmp = f['Labels'][:limit,...] if limit else f['Labels'][...]
        n_f.create_dataset('Images', data = ds, dtype = np.float32)
        n_f.create_dataset('Labels', data = tmp, dtype = np.uint8)
    else:
        print "%s has NaN after conversion" %fn
    n_f.close()


    
if __name__ == "__main__":
    if len(sys.argv)>1:
        ## make a special file
        limit = int(sys.argv[2]) if len(sys.argv)>2 else None
        convert_sample(sys.argv[1], limit)
    else:
        fl = []
        fl.extend(glob.glob('/bigdata/shared/Delphes/np_datasets_new/3_way/MaxLepDeltaR_des/train/*.h5'))
        fl.extend(glob.glob('/bigdata/shared/Delphes/np_datasets_new/3_way/MaxLepDeltaR_des/val/*.h5'))
        random.shuffle( fl )
        every = 5
        N= None
        for i,fn in enumerate(fl):
            com = 'python Transform.py %s'%( fn)
            if N: com += ' %d'%N
            wait = (i%every==(every-1))
            if not wait: com +='&'
            print com
            os.system(com)
            if wait and N:
                time.sleep( 60 )

