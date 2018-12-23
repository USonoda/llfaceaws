a = [
    ['kotori', 141, 25],
    ['hanayo', 120, 21],
    ['maki', 128, 23],
    ['others', 392, 69],
    ['rin', 132, 23],
    ['nozomi', 135, 24],
    ['nico', 138, 24],
    ['umi', 121, 21],
    ['eli', 119, 21],
    ['honoka', 119, 21]
]
te, va = 0,0
for i in range(10):
    te += a[i][1]
    va += a[i][2]
print('test:'+str(te)+' valid:'+str(va))
# test:2736(移動前) valid:482(75)
# test:1545 valid:272 (150)
