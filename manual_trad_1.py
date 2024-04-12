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
            



curmax = 0
for y1 in range(900,1001):
    for y2 in range(900,1001):
        if y2 > y1:
            profit = (y1**2-900**2)/(1000**2-900**2) * (1000-y1) + (1-((y1**2-900**2)/(1000**2-900**2))) * (y2**2-y1**2)/(1000**2-900**2) * (1000-y2)
            if profit >= curmax:
                curmax = profit
                maxy1 = y1
                maxy2 = y2
                print(f"maxi={y1},maxj={y2},profit={profit}")   
            