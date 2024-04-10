curmax = 0
for i in range(900,1001):
    for j in range(900,1001):
        if j > i:
            profit = (i**2)/190 - (900**2)/190 - (i**3)/190000 + (900**2)*i/190000 + (j**2)/190 - (i**2)/190 - (j**3)/190000 + (i**2)*j/190000
            if profit >= curmax:
                curmax = profit
                maxi = i
                maxj = j
                print(f"maxi={i},maxj={j},profit={profit}")   
            
        