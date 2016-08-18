from hparams import HParams

hparams = HParams(a=1, b=0.1, c=True, d=[1,2.2, 4,5])
assert hparams.a == 1
assert hparams.b == 0.1
assert hparams.c == True
assert hparams.d == [1, 2.2,4, 5]

print "Passed test 1"

flag = "a=1000, b=0.124, c=False,d=[1, 5.6,23, 0]"

hparams.parse(flag)
assert hparams.a == 1000
assert hparams.b == 0.124
assert hparams.c == False
assert hparams.d == [1,5.6,23,0]

print "Passed test 2"

try:
  hparams.parse("a=False")
except ValueError:
  print "Passed test 3"

try:
  hparams.parse("b=11.3.4")
except ValueError:
  print "Passed test 4"

try:
  hparams.parse("c=false")
except ValueError:
  print "Passed test 5"

try:
  hparams.parse("d=1")
except ValueError:
  print "Passed test 6"


