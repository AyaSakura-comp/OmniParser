import unittest,importlib.util
import torch
class AdapterTests(unittest.TestCase):
 def test_adapter_exists(self):
  self.assertIsNotNone(importlib.util.find_spec('cpu_adapter'),'CPU adapter not implemented')
 def test_fp32_reference_is_not_mutated(self):
  if importlib.util.find_spec('cpu_adapter') is None:self.skipTest('adapter missing')
  from cpu_adapter import prepare
  m=torch.jit.trace(torch.nn.Sequential(torch.nn.Conv2d(3,4,3,padding=1),torch.nn.BatchNorm2d(4)).eval(),torch.randn(1,3,32,32))
  before=[p.clone() for p in m.parameters()]
  prepared=prepare(m)
  for old,new in zip(before,m.parameters()):self.assertTrue(torch.equal(old,new));self.assertEqual(new.dtype,torch.float32)
  x=torch.randn(1,3,32,32)
  with torch.inference_mode():a=m(x);b=prepared(x.bfloat16())
  self.assertEqual(b.dtype,torch.bfloat16)
  self.assertGreater(torch.nn.functional.cosine_similarity(a.flatten(),b.float().flatten(),dim=0),.99)
 def test_forward_returns_fp32_and_rejects_wrong_shape(self):
  import cpu_adapter
  self.assertTrue(hasattr(cpu_adapter,'CPUForward'),'CPUForward not implemented')
  model=torch.jit.trace(torch.nn.Conv2d(3,4,3,padding=1).eval(),torch.randn(1,3,32,32))
  forward=cpu_adapter.CPUForward(model,32)
  with torch.inference_mode():out=forward(torch.randn(1,3,32,32))
  self.assertEqual(out.dtype,torch.float32)
  with self.assertRaises(ValueError):forward(torch.randn(1,3,64,64))
if __name__=='__main__':unittest.main()
