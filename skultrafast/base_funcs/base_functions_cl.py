import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

mf = cl.mem_flags
#PYOPENCL_COMPILER_OUTPUT=1
plat = cl.get_platforms()[-1]
dev = plat.get_devices()[0]
ctx = cl.Context([dev])
queue = cl.CommandQueue(ctx)
work_size = (128, 1)


def use_deriv(eg=None):
    if not deg:
        deg = [1, 2, 3, 4]


pre = """
#define SQRT_2PI 2.5066282746310002f
 __kernel void coh_gauss(__global const float *t, const float w,
                         __global float *z, const int max_gid)
    {
      int gid = get_global_id(0);
      if (gid > max_gid) {return;};
      int gid2 = get_global_id(1);
      int len_tau = get_global_size(1);
      int idx = gid * len_tau + gid2;
      
      float tt = t[gid];
      
      
      if ((t[gid] < -w*4.) || (w*4. < t[gid])){z[idx] = 0.;}
          {
              z[idx] = native_exp(-.5f * (tt / w) * (tt / w)) ;
              if (gid2 == {0}) {z[idx] *= (-tt / w / w); };
              if (gid2 == {}) {z[idx] *= (tt * tt / w / w  / w / w - 1 / w /w);};
              if (gid2 == 3) {z[idx] *= (-pow(tt, 3) / pow(w, 6) + 3. * tt / pow(w,4));};
          }
    }
"""





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
                        __global const float *k,  __global float *z, 
                        const int max_gid)
    {
      int gid = get_global_id(0);
      int gid2 = get_global_id(1);
      int len_tau = get_global_size(1);
      int idx = gid * len_tau + gid2;

      if (gid > max_gid) {return;};
      
      if (t[gid]< -w*2.5){z[idx] = 0;}
          {z[idx] = native_exp(k[gid2]*(w*w*k[gid2]*.25f - t[gid]));};
      if (t[gid]< 2.5 * w) {z[idx] *= .5* erfc(-t[gid]/w + w*k[gid2]*0.5);};    
    }
""").build()


prg2 = cl.Program(ctx, """
#define SQRT_2PI 2.5066282746310002f
 __kernel void coh_gauss(__global const float *t, const float w,
                         __global float *z, const int max_gid)
    {
      int gid = get_global_id(0);
      if (gid > max_gid) {return;};
      int gid2 = get_global_id(1);
      int len_tau = get_global_size(1);
      int idx = gid * len_tau + gid2;
      
      float tt = t[gid];
      
      
      if ((t[gid] < -w*4.) || (w*4. < t[gid])){z[idx] = 0.;}
          {
              z[idx] = native_exp(-.5f * (tt / w) * (tt / w)) ;
              if (gid2 == 1) {z[idx] *= (-tt / w / w); };
              if (gid2 == 2) {z[idx] *= (tt * tt / w / w  / w / w - 1 / w /w);};
              if (gid2 == 3) {z[idx] *= (-pow(tt, 3) / pow(w, 6) + 3. * tt / pow(w,4));};
          }
    }
""").build()

coh_no_div = cl.Program(ctx, """
#define SQRT_2PI 2.5066282746310002f
 __kernel void coh_gauss(__global const float *t, const float w,
                         __global float *z, const int max_gid)
    {
      int gid = get_global_id(0);
      if (gid > max_gid) {return;};
      int gid2 = get_global_id(1);
      int len_tau = get_global_size(1);
      int idx = gid * len_tau + gid2;
      
      float tt = t[gid];
      
      
      if ((t[gid] < -w*4.) || (w*4. < t[gid])){z[idx] = 0.;}
          {
              z[idx] = native_exp(-.5f * (tt / w) * (tt / w)) ;
              if (gid2 == 1) {z[idx] *= (-tt / w / w); };              
              if (gid2 == 2) {z[idx] *= (tt * tt / w / w  / w / w - 1 / w /w);};
              
          }
    }
""").build()



prg3 = cl.Program(ctx, """

__kernel void multi_dot(__global const float *model, 
                        __global const float *coeff, 
                        __global float *out, 
                        const int n, const int m)
    {
      int y = get_global_id(0);
      int x = get_global_id(1);
      int sizeX = get_global_size(1);
      int sizeY = get_global_size(0);
      
      int m_idx;
      
            
      for (int i=0; i<n; i++)           
          {
          m_idx = i * sizeX * sizeY + sizeX * y + x;        
          //printf("i %d x %d y %d %f %d\\n", i,x, y,  model[m_idx], m_idx);
          //printf("coeff %d %f\\n", x + sizeX * i, coeff[x*n + i]);
          out[x + y * sizeX] += model[m_idx] * coeff[x + sizeX * i];};
    }
    
""").build()



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

def _coh_gaussian2(t_array, w, tz):
    if tz != 0.:
        t_array -= tz

    shape = t_array.shape
    t_array = t_array.astype(np.float32)
    t_arr_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_array)
    shape = (shape[0], shape[1], 4)
    out = cl_array.empty(queue, shape=shape, dtype=np.float32)

    prg2.coh_gauss(queue, (t_array.size, 3), None, t_arr_gpu,  np.float32(w/1.4142), out.data).wait()

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
    global_work_size = t_array.size + (work_size[0] - t_array.size % work_size[0])
    prg.fold_exp(queue, (global_work_size, tau_arr.size), work_size, t_arr_gpu, np.float32(w),
                 tau_buf, out.data, np.uint32(t_array.size)).wait()

    a = out.get()

    return a

def _fold_exp_and_coh(t_array, w, tz, tau_arr):
    if tz != 0.:
        t_array -= tz

    shape = t_array.shape
    t_array = t_array.astype(np.float32)

    t_arr_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_array)
    tau_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=(1/tau_arr).astype(np.float32))
    shape = (shape[0], shape[1], tau_arr.size)
    shape_coh = (shape[0], shape[1], 3)
    out = cl_array.empty(queue, shape=shape, dtype=np.float32)
    out_coh = cl_array.empty(queue, shape=shape_coh, dtype=np.float32)

    global_work_size = t_array.size + (work_size[0] - t_array.size % work_size[0])

    prg.fold_exp(queue, (global_work_size, tau_arr.size), work_size, t_arr_gpu, np.float32(w),
                 tau_buf, out.data, np.uint32(t_array.size))

    coh_no_div.coh_gauss(queue, (global_work_size, 3), work_size, t_arr_gpu,
                   np.float32(w/1.4142), out_coh.data, np.uint32(t_array.size))

    queue.finish()
    a = out.get(async_=True)
    b = out_coh.get(async_=True)
    b /= np.abs(b).max(0)
    queue.finish()
    return a, b

t_array = np.subtract.outer(np.linspace(-1, 5, 128*4),
                            np.linspace(3, 3, 256*4))


#t_array = np.add.outer(np.arange(10, 50, 10),
#                            np.arange(1, 3))
#
#print t_array
#a,b = _fold_exp_and_coh(t_array, 0.1, 0., np.array([1., 10., 50., 300., 1000.]))
#print a.shape
##a = t_array[:, :, None]
#c = np.random.rand(*a.shape[1:])
##c = np.ones(a.shape[1:])
##c.flat[1] = 2
#a_gpu = cl_array.to_device(queue, a.astype(np.float32))
##c_gpu = cl_array.to_device(queue, c.astype(np.float32))
##out = cl_array.zeros(queue, shape=(a.shape[0], a.shape[1]), dtype=np.float32)
###print a
##prg3.multi_dot(queue, out.shape, None, a_gpu.data, c_gpu.data,
##               out.data, np.uint32(a_gpu.shape[-1]), np.uint32(30)).wait()
##ax = out.get()
#
def multi_dot(a_gpu, c):
    #a_gpu = cl_array.to_device(queue, a.astype(np.float32))
    c_gpu = cl_array.to_device(queue, c.astype(np.float32))
    out = cl_array.empty(queue, shape=(a_gpu.shape[0], a_gpu.shape[1]), dtype=np.float32)

    prg3.multi_dot(queue, out.shape, (128,1), a_gpu.data, c_gpu.data,
               out.data, np.uint32(a_gpu.shape[-1]), np.uint32(30)).wait()
    ax = out.get()
    return ax

#import time
#from numpy.core.umath_tests import inner1d
#t = time.time()
#multi_dot(a_gpu, c)
#print (time.time() - t)*1000.
#
#t = time.time()
#inner1d(a, c)
#print (time.time() - t)*1000.
#
#
##
#
#
#t_array = np.subtract.outer(np.linspace(-1, 5, 601),
#                            np.linspace(3, 3, 401))

#from skultrafast.base_functions import _fold_exp as _fold_exp2
#from skultrafast.base_functions import _coh_gaussian as coh_g2
#a,b = _fold_exp_and_coh(t_array, 0.1, 0., np.array([1., 10. ,1000., 10]))
##coh = _coh_gaussian(t_array, 0.5, 0.)
##coh2 = coh_g2(t_array, 0.5, 0)
