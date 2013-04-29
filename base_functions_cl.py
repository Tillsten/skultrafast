import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array    

mf = cl.mem_flags
#PYOPENCL_COMPILER_OUTPUT=1
plat = cl.get_platforms()[1]
dev = plat.get_devices()[0]
ctx = cl.Context([dev])
queue = cl.CommandQueue(ctx)

prg = cl.Program(ctx, """
/* Please Write the OpenCL Kernel(s) code here*/
#define a1  .278393f
#define a2  .230389f
#define a3  .000972f
#define a4  .078108f
float inline fast_erfc(float x)
    {
    uint smaller = 0;
    float bot; 
    float ret;
    if (x<0) 
        {
         x = x * -1.;
         smaller = 1;
         }
    bot = 1 + x *a1 + x*x * a2 +x*x*x * a3 + x*x*x*x * a4;
    ret = 1./(bot*bot*bot*bot);

    if (smaller)  {return -ret + 2.;}  {return  ret;}
    }    
    
 __kernel void fold_exp(__global const float *t, const float w, 
                        __global const float *k,  __global float *z)
    {
      int gid = get_global_id(0);
      int gid2 = get_global_id(1);
      int len_tau = get_global_size(1);
      int idx = gid * len_tau + gid2;
      
      if (t[gid]< -w*2.5){z[idx] = 0;}
          {z[idx] = native_exp(k[gid2]*(w*w*k[gid2]*.25 - t[gid]));};
      if (t[gid]< 2.5 * w) {z[idx] *= .5* erfc(-t[gid]/w + w*k[gid2]*0.5);};    
    }
""").build()


prg2 = cl.Program(ctx, """
#define SQRT_2PI 2.5066282746310002f
 __kernel void coh_gauss(__global const float *t, const float w, __global float *z)
    {
      int gid = get_global_id(0);
      int gid2 = get_global_id(1);
      int len_tau = get_global_size(1);
      int idx = gid * len_tau + gid2;
      float tt = t[gid];
      
      if ((t[gid] < -w*4.) || (w*4. < t[gid])){z[idx] = 0.;}
          {
              z[idx] = exp(-.5 * (tt / w) * (tt / w)) ;
              if (gid2 == 1) {z[idx] *= (-tt / w / w); };
              if (gid2 == 2) {z[idx] *= (tt * tt / w / w  / w / w - 1 / w /w);};
              if (gid2 == 3) {z[idx] *= (-pow(tt, 3) / pow(w, 6) + 3. * tt / pow(w,4));};
          }
    }
""").build()
#                y[i, j, 0] = np.exp(-0.5 * (tt / w)* (tt / w)) / (w * np.sqrt(2 * 3.14159265))
#                y[i, j, 1] = y[i, j, 0] * (-tt / w / w)
#                y[i, j, 2] = y[i, j, 0] * (tt * tt / w / w  / w / w - 1 / w /w)
#                y[i, j, 3] = y[i, j, 0] * (-tt ** 3 / w ** 6 + 3 * tt / w ** 4)

def _coh_gaussian(t_array, w, tz):
    if tz != 0.:
        t_array -= tz
    
    shape = t_array.shape    
    t_array = t_array.astype(np.float32)    
    t_arr_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_array)
    shape = (shape[0], shape[1], 4)
    out = cl_array.empty(queue, shape=shape, dtype=np.float32)  
    
    prg2.coh_gauss(queue, (t_array.size, 4), None, t_arr_gpu,  np.float32(w/1.4142), out.data).wait()    
    
    a = out.get()
    a /= np.abs(a).max(0)
    return a

def _fold_exp(t_array, w, tz, tau_arr):    
    if tz != 0.:
        t_array -= tz
    
    shape = t_array.shape    
    t_array = t_array.astype(np.float32)
    
    t_arr_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_array)
    tau_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                        hostbuf=(1/tau_arr).astype(np.float32))
    shape = (shape[0], shape[1], tau_arr.size)
    out = cl_array.empty(queue, shape=shape, dtype=np.float32)    
    prg.fold_exp(queue, (t_array.size, tau_arr.size), None, t_arr_gpu, np.float32(w), 
                 tau_buf, out.data).wait()    
    
    a = out.get()

    return a

t_array = np.subtract.outer(np.linspace(-1, 5, 600),
                            np.linspace(3, 3, 400))

from skultrafast.base_functions import _fold_exp as _fold_exp2
from skultrafast.base_functions import _coh_gaussian as coh_g2
#a = _fold_exp(t_array, 0.1, 0., np.array([1., 10. ,1000., 10]))
coh = _coh_gaussian(t_array, 0.5, 0.)
coh2 = coh_g2(t_array, 0.5, 0)