[[EXIST project]]

In this document, we verifies whether our proposed invariant I satisfy 
`I = [loop guard] · wp(program, I) + [¬loop guard] · post`. 

## Geo0
Proposed invariant **I** is

	wp(Geo0, z) = [flip != 0] * z + [flip = 0] * (z + (1 - p)/p)
	
**Verified**:
Let I = z + [flip = 0] * (1 - prob)/prob

	wp(body, I) = p * wp(flip = 1, I) + (1-p) * wp(z = z+1, I)
	= p * (z + 0) + (1-p) * (z + 1 + [flip = 0] * (1 - p)/p)) 
	= z + (1-p) + [flip = 0] * (1-p) * (1 - p)/p
	= [flip != 0] (z + 1-p) + [flip = 0] * (z + 1-p + (1-p) * (1 - p)/p)
	= [flip != 0] (z + 1-p) + [flip = 0] * (z + (1-p)/p)

Thus, 

	[flip = 0] *wp(body, I) + [flip != 0] * post
	= [flip = 0] * (z + (1-p)/p) + [flip != 0] z 
	= I

## Geo1
Proposed invariant **I** is

	wp(Geo1, z) = [flip != 0] * z + [flip = 0] * (z + (1 - p)/p)
	
**Verification Skipped**

## Geo0 
Proposed invariant **I** is

	wp(Geo2, z) = [flip != 0] * z + [flip = 0] * (z + (1 - p)/p)
	
**Verification Skipped**

## Fair
Proposed invariant **I** is

	wp(Fair, count) = count + [not (c1 or c2)] * (p1 + p2) / (p1 + p2 - p1 * p2)

**Verified:**

	wp(body, I) = wp({count = count + 1}[c1 = bernoulli(p1)]{skip}, 
	p2 * (count + 1 + [not (c1 or 1)] * (p_1 + p_2) / (p_1 + p_2 - p_1 * p_2)) + 
	(1-p2) * (count + [not (c1 or 0)] * (p_1 + p_2) / (p_1 + p_2 - p_1 * p_2))) 
	= wp({count = count + 1}[c1 = bernoulli(p1)]{skip}, 
	p2 * (count + 1) + (1-p2) * (count + [not (c1)] * (p_1 + p_2) / (p_1 + p_2 - p_1 * p_2))) 
	= p1 * (p2 * (count + 2) + (1-p2) * (count + 1))
	+ (1-p1) * (p2 * (count + 1) + (1-p2) * (count + (p_1 + p_2) / (p_1 + p_2 - p_1 * p_2)))) 
	= count + 2p1p2 + p1(1-p2) + (1-p1)p2 + (1-p1)(1-p2)(p_1 + p_2) / (p_1 + p_2 - p_1 * p_2)
	= count + p1 + p2 + (1-p1-p2+p1p2)(p_1 + p_2) / (p_1 + p_2 - p_1 * p_2)
	= count + (p_1 + p_2) / (p_1 + p_2 - p_1 * p_2) 

Thus, 

	[not (c1 or c2)] * wp(body, I) = [c1 or c2] * count = I

## Mart
Proposed invariant **I** is

	 wp(Mart, rounds) = rounds + [b > 0] * (1/p)

**Verified:**
	
	wp(body,I) = wp({...}[p]{...}, rounds + 1 + [b > 0] * (1/ p))
	 = p * (rounds + 1) + (1-p) * (rounds + 1 + [2b > 0] * (1/ p))
	 = rounds + 1 + [b > 0] * ((1 - p) / p)
	 
and 

	 [b > 0] * wp(body,I) 
	 = [b > 0] * (rounds + 1 + (1 - p)/ p)
	 = [b > 0] * (rounds + 1/p)

 Thus, `[b > 0] * wp(body,I) + [b <= 0] * post = rounds + [b > 0] * (1/p) = I`.

## Gambler0
Proposed invariant **I** is
	
	wp(Gambler0, z) = z + [0 < x and x < y] * x * (y - x)

**Verified**

	wp(body of Gambler0, I)
	= p * (z + 1 + [0 < x + 1 < y] * (x + 1) * (y - x - 1)) + (1-p) * (z + 1 + [0 < x - 1 < y] * (x - 1) * (y - x + 1))
	= z + 1 + [0 < x + 1 < y] * p * (x + 1) * (y - x - 1) + [0 < x - 1 < y] * (1-p) * (x - 1) * (y - x + 1)
	
Then, 

	[G]* wp(body, I) = [0 < x < y](z + 1 + [0 < x-1 < x + 1 < y] * (x * (y - x) - 1) + 
	[x = 1] * p * (x + 1) * (y - x - 1)  + [x = y - 1] * (1-p) * (x - 1) * (y - x + 1))
	
Recall that p = 0.5. Then, the expression equals to 

	 [0 < x < y](z + 1 + [0 < x-1 < x + 1 < y] * (x * (y - x) - 1) + 
	[x = 1] * (y - x - 1)  + [x = y - 1] * (x - 1)
	= [0 < x < y](z + [0 < x-1 < x + 1 < y] * (x * (y - x) ) + [x = 1] * x * (y - x)  + [x = y - 1] * (x - 1) * (y-x))
	= [0 < x < y](z + (x * (y - x))
	
Thus, 

	[G]* wp(body, I) + [not G] * z = z + [0 < x and x < y] * x * (y - x) 

**Our (CAV) algorithm did wrong on this example. **

Our algorithm learned 
	
	E = z + [x * (y - x) <= 1.000]( 0 < x < y) + [x * (y - x) > 1.000](x * (y - x)). 
It's incorrect because 

	[G] * wp(Gambler0 body, E) + [not G] * z = I != E. 

We proves the claim in two steps: 
1. `I != E` because 
	`[not G] * I = [0 >= x or x >= y] * z`, 
	`[not G] * E = [0 >= x or x >= y] * (z + [x * (y - x) > 1.000] * x * (y - x))`
For instance, consider x = -2, y = -4, z = 0, then x * (y - x) = 4 > 1, 
and `[not G] * I` evaluates to 0, and `[not G] * E` evaluates to 4. 

2. `[G] * wp(Gambler0 body, E) + [not G] * z = I` because 

		wp(body, E)
		= wp(body,  z + [x * (y - x) <= 1.000]( 0 < x < y) + [x * (y - x) > 1.000](x * (y - x)) ) 
		= wp(..[p]...,  z + 1 + [x * (y - x) <= 1.000]( 0 < x < y) + [x * (y - x) > 1.000](x * (y - x)) ) 
		= z + 1 + [(x + 1) * (y - x - 1) <= 1.000]( 0.5 * (0 < x + 1 < y)) + [(x + 1) * (y - x - 1) > 1.000](0.5 * (x + 1) * (y - x - 1)) + [(x - 1) * (y - x + 1) <= 1.000]( 0.5 * (0 < x - 1 < y)) + [(x - 1) * (y - x + 1) > 1.000](0.5 * (x - 1) * (y - x + 1)) )
	
Note that 
		
		(x + 1) * (y - x - 1) <= 1.000 is equivalent to x * (y - x) + y - 2x - 2 <= 0
		(x - 1) * (y - x + 1) <= 1.000 is equivalent to x * (y - x) - y + 2x - 2 <= 0

When `0 < x < y, x * (y-x) > 0`, since they are all integers, we have y >= x + 1. Then we can do case analysis: 

	-- If y >= x + 2 and x - 1 >= 1 then 
		(x + 1) * (y - x - 1) >= 3 * 1 = 3
		(x - 1) * (y - x + 1) >= 1 * 3 = 3 
		so G * wp(body, E)
		= [0 < x < y](z + 1 + 0.5 * (x + 1) * (y - x - 1) + 0.5 * (x - 1) * (y - x + 1))
		= [0 < x < y](z + 1 + x * (y - x) - 1)  
		= [0 < x < y](z + x * (y - x))  
		G * wp(body, E) + not G * post = I but not equals what we learned
	-- If y >= x + 2 and 0 < x < 2, then it must x = 1, 
		(x + 1) * (y - x - 1) >= 2 * 1 = 2
		(x - 1) * (y - x + 1) <= 0 * positive = 0 
		so G * wp(body, E)
		= [0 < x < y](z + 1 + 0.5 * (x + 1) * (y - x - 1) + 0.5 * (0 < x - 1 < y))
		= [0 < x < y](z + y - x) 
		= [0 < x < y](z + x (y - x))
	-- If y < x + 2 and x - 1 >= 1, then it must y = x + 1
		(x + 1) * (y - x - 1) = pos * non pos = non pos
		(x - 1) * (y - x + 1) >= 1 * 2 = 2
		so G * wp(body, E)
		= [0 < x < y](z + 1 + 0.5 * (0 < x + 1 < y)+ 0.5 * (x - 1) * (y - x + 1))
		= [0 < x < y](z + 1 + 0.5 * (x - 1) * 2)
		= [0 < x < y](z + x)
		= [0 < x < y](z + x * (y-x)) 
	-- If y <= x + 1 and 0 < x < 2, then it must y = x + 1, x =1
		(x + 1) * (y - x - 1) = 2 * non pos = non pos
		(x - 1) * (y - x + 1) = 0
		so G * wp(body, E) 
		= [0 < x < y](z + 1 + 0.5 * (0 < x + 1 < y) + 0.5 * (0 < x - 1 < y))
		= [0 < x < y](z + 1 + 0.5 * (0 < x + 1 < y) )
		= [0 < x < y](z + 1) 
		= [0 < x < y](z + x * (y-x)) 

## GeoAr0
Proposed invariant **I** is

	wp(GeoAr0, x) =  [z = 0]* x + [z != 0] (x + y(1-p)/p + (1-p)/(p**2))

**Verified**

	wp(body, I)
	= wp(y=y+1, p * I[z/0] + (1-p) * I[x <- x + y])
	= wp(y=y+1, p * x + (1-p) * ([z = 0]* (x + y) + [z != 0] (x +y+ y(1-p)/p + (1-p)/(p^2))) )
	=  p * x + (1-p) * ([z = 0]* (x + y + 1) + [z != 0] (x + y +1 + y(1-p)/p + (1-p)/p + (1-p)/(p^2))) 
	=  p * x + (1-p) * ([z = 0]* (x + y + 1) + [z != 0] (x + y +1 + y(1-p)/p  + (1-p^2)/(p^2))) 

Then, 

	[G] * wp(I, body) 
	= p * x + (1-p) * ([z != 0] (x + y +1 + y(1-p)/p  + (1-p^2)/(p^2))) 
	= [z != 0] (x + y(1-p)/p + (1-p)/(p^2))

So `[G] * wp(body, I)  + [not G] * z = I.`

## GeoAr1
Proposed invariant **I** is

	wp(GeoAr1, x) = [z = 0]* x + [z != 0] (x + 12)
		
**Verification Skipped**

## GeoAr2
Proposed invariant **I** is 

	wp(GeoAr2, x) =  [z = 0]* x + [z != 0] (x + 3y + 12)

**Verification Skipped**


## GeoAr3
Proposed invariant **I** is

	wp(GeoAr3, x) =  [z = 0]* x + [z != 0] (x + (1-p)/(p**2))

**Verification Skipped**


## Bin0
Proposed invariant **I** is
	
	wp(Bin0, x) = [n > 0] * (x + p * n * y) + [n <= 0] * x

**Verification Skipped**

## Bin1
Proposed invariant **I** is

	wp(Bin1, x) = [n − M < 0] * (x - p*n + p*M) + [n − M >= 0] * x
	
**Verification Skipped**

**Our (CAV) algorithm did wrong on this example. **

Our algorithm learned `E =  [n − M <= 0] * (x - p*n + p*M)`. Note that 

	wp(body,E)
	=  (p)[n+1 − M <= 0] * (x + 1 - p*(n+1) + p*M) + (1-p)[n+1 − M <= 0] * (x - p*(n+1) + p*M)
	=  [n − M <= -1] * (x + p - p*(n+1) + p*M) 
	=  [n − M <= -1] * (x - p*(n) + p*M) 
	
Thus, 

	[G] * E + [not G] * post 
	=  [n − M <= -1] * (x - p*(n) + p*M) + [n - M >= 0] * x 
	!= E
because when n - M > 0, they evaluate to different things. 

## Bin2 
Proposed invariant **I** is 
	
	wp(Bin2,x) = x + [n > 0] * (0.5pn(n+1) + (1-p)ny)

**Verified**

	wp(Bin2, I)
	= wp(block, x + [n-1 > 0] * (0.5p*(n-1) * (n) + (1-p) * (n - 1) * y))
	= p * (x + n + [n-1 > 0] * (0.5p*(n-1) * (n) + (1-p) * (n - 1) * y)) + 
	(1-p) * (x + y + [n-1 > 0] * (0.5p*(n-1) * (n) + (1-p) * (n - 1) * y)) 
	= x + p * n + (1-p) * y + [n-1 > 0] * (0.5p*(n-1) * (n) + (1-p) * (n - 1) * y) 

	[G] * wp(Bin2, I)
	= [n > 0] * (x + p * n + (1-p) * y + [n-1 > 0] * (0.5p*(n-1) * (n) + (1-p) * (n - 1) * y) )
	= [n > 0] * (x + [n = 1] * (p * n + (1-p) * y) + 
		[n-1 > 0] * (p * n + (1-p) * y + 0.5p*(n-1) * (n) + (1-p) * (n - 1) * y)
	= [n > 0] * (x + [n = 1] * (p * n + (1-p) * y) + 
		[n-1 > 0] * (0.5p*(n+1) * (n) + (1-p) * n * y))
	= [n > 0] * (x + [n = 1] * (p * n + (1-p) * y) + 
		[n-1 > 0] * (0.5p*(n+1) * (n) + (1-p) * n * y))
	=  [n > 0] * (0.5pn(n+1))

Thus, `[G] * wp(Bin2, I) + [not G] * post = I`. 

## Bin3
Proposed invariant **I** is
	
	wp(Bin3,x) = [n > 0] * 0.125n(n+1) + 0.75ny
	
**Verification Skipped**

## LinExp
Proposed invariant **I** is

	wp(LinExp, count) = [n<=0] * count + [n>0] * (count + n*(7*3/8))
	
**Verification Skipped**

## Seq0
Proposed invariant **I** is

	wp(Seq0, x) = x + [flip1 = 0] * p1/(1-p1) + [flip2 = 0] * (p2/(1-p2))
	
**Verification Skipped**

## Seq1
Proposed invariant **I** is
	
	wp(Seq1, x) = x + [flip1 = 0] * p1/(1-p1) + [flip2 = 0] * (-p2/(1-p2))

**Verification Skipped**

## Nest
Proposed invariant **I** is

	wp(Nest, x) = x + [flip1 = 0 and flip2 = 0] * p1/(1-p1) * p2/(1-p2) + [flip1 = 0 and flip2 != 0] * p1 * p1 /(1-p1) * p2/(1-p2)

**Verified:**

	wp(body, I)
	= p1*wp(loop1, x + [flip1 = 0 and 0 = 0] * p1/(1-p1) * p2/(1-p2) 
		   + [flip1 = 0 and 0 != 0] * p1 * p1 /(1-p1) * p2/(1-p2)
	+(1-p1)*x
	= p1*wp(loop1, x+ [flip1 = 0] * p1/(1-p1) * p2/(1-p2))) +(1-p1)*x
	= p1* (x + [flip2 = 0] p2/(1-p2) + [flip1 = 0] * p1/(1-p1) * p2/(1-p2))) 
	+(1-p1)*x
	= x + [flip2 = 0] p2/(1-p2) * p1 + [flip1 = 0] * p1/(1-p1) * p2/(1-p2) * p1

Then, 

	wp(body, I) * [flip1 = 0]
	= [flip1 = 0] (x + [flip2 = 0] p2/(1-p2) * p1 + p1/(1-p1) * p2/(1-p2) * p1)
	= [flip1 = 0] (x + [flip2 = 0] (p2/(1-p2) * p1 + p1/(1-p1) * p2/(1-p2) * p1) 
	+ [flip2 != 0](p1/(1-p1) * p2/(1-p2) * p1)) 
	= [flip1 = 0] * x + [flip1 = 0 and flip2 = 0] (p1 * p1 /(1-p1) * p2/(1-p2)) 
	+ [flip1 = 0 and flip2 != 0](p1/(1-p1) * p2/(1-p2) * p1))   

Thus, `wp(body, I) * [flip1 = 0] + x * [flip1 != 0] = I`. 

## Sum0
Proposed invariant **I** is

	wp(Sum0, x) = [n > 0] * (x + p * (0.5n(n+1))) + [n <= 0] * x

**Verification Skipped**

## Sum1
Proposed invariant **I** is
	
	wp(Sum1, x) specializes wp(Sum0, x) at p = 0.25

**Verification Skipped**

## DepRV
Proposed invariant **I** is

	wp(DepRV, xy) ([n>0]*(1/4(n^2 + 2nx + 2ny + 4xy - n))+ [n<=0]*(xy)
	
**Verification Skipped**

## BiasPri
Proposed invariant **I** is

	wp(BiasPri, [x = 0 and y − 1 = 0] − [x − 1 = 0 and y = 0])
	= [x = 0 and y − 1 = 0] − [x − 1 = 0 and y = 0]

**Verified: **

	wp(body, I) 
	 = wp(first block, p * -[x − 1 = 0 and 0 = 0] + (1-p) * [x = 0 and 1 -1 =0])
	 = wp(...[p]..., p * -[x − 1 = 0] + (1-p) * [x = 0])
	 = (1-p) * p * -[1 − 1 = 0] + p * (1-p) * [0 = 0]
	 = p * (1-p) * (-1 + 1) 
	 = 0

Thus, 

	(x-y == 0) * wp(body, I) + [x = 0 and y − 1 = 0] − [x − 1 = 0 and y = 0]
	= [x = 0 and y − 1 = 0] − [x − 1 = 0 and y = 0]
	
So I is an invariant.  

## BiasDir
Proposed invariant **I** is

	wp(BiasDir, x) = [x != y] * x + [x = y] * 1/2

**Verified:**

	wp(body, I) 
	 = wp(first block, p * ([x != 0] * x + [x = 0] * 1/2) + (1-p) * ([x != 1] * x + [x = 1] * 1/2))
	 = wp(first block, [x = 1] * (p * x + (1-p)/2) + [x = 0] * (p/2 + (1-p) * x))
	 = (1-p) * [1 = 1] * (p * 1 + (1-p)/2) + p * [0 = 0] * (p/2 + (1-p) * 0) 
	 = (1-p) * p + (1-p) * (1-p)/2 + p * p/2 
	 = 1/2

Then, 

	[G] * wp(body, I) + [not G] * post 
	= [x != y] * x + [x = y] * 1/2

## Prinsys 
Proposed invariant **I** is

	wp(Prinsys, x=1) = [x = 0] * (1-p2) + [x != 0] * [x = 1]
	
**Verified:**

	wp(body, I) = p1 * (1-p2) + (1-p1) * (p2*0 + (1-p2)*1)
	= 1 - p2

Thus, `[G] * wp(body, I) + [not G] * I = I`.

## Duel
Proposed invariant **I** is

	wp(Duel, t) = [t = A and c = 0]+ [t = A and c = 1] * (p1/(p1 + p2 - p1 * p2))+ \
							   [t = B and c = 1] * (1 - p2) * p1/(p1 + p2 - p1 * p2)

**Verified:**

Since we let t = True if t = A, t = False if t = B, 
	
	I = [t = True and c = 0]+ [t = True and c = 1] * (p1/(p1 + p2 - p1 * p2))+ \
 			[t = False and c = 1] * ((1 - p2) * p1/(p1 + p2 - p1 * p2))

	wp(Duel, I)
	= [t = True] * wp(block1, [c = 0]+ [c = 1] * p1/(p1 + p2 - p1 * p2) )+ 
	 [t = False] * wp(block2, [c = 1] * (1 - p2) * p1/(p1 + p2 - p1 * p2) )
	= [t = True] * (p1 * [0 = 0]) 
	+ [t = False] * ([c = 0] * (1-p1) + [c=1] * (1-p1) * p1/(p1 + p2 - p1 * p2))

	[G] * wp(Duel, I)
	= [c = 1] * ([t = True] * p1 + [t = False] * [c=1] * (1-p1) * p1/(p1 + p2 - p1 * p2)) 
	= [c = 1 and t = True] * p1 + [c = 1 and t = False] * [c=1] * (1-p1) * p1/(p1 + p2 - p1 * p2)) 

So `[G] * wp(Duel, I) + [not G] * post = I`, and thus I is an invariant. 


## Unif 
Proposed invariant **I** is

	wp(Unif, count) = count + [x <= 10]*(10-x+1)

**Verified:**

	wp(body, I) = wp(x= x + randint(0,2), count+1 +[x <= 10]*(10-x+1))
	= E_[v in [0,2]] count+1 + [x <= 10]*(10-x-v+1)
	= count+1 + [x <= 10]*(10-x-1+1) 
	= count+1 + [x <= 10]*(10-x) 

Thus, 

	[x <= 10] * wp(body, I) = [x <= 10]*(count+1+10-x)
	
And `[x <= 10] * wp(body, I) + [x > 10] * post = I `

## Detm
Proposed invariant **I** is

	wp(Detm, count) = count + [x <= 10]*(10-x+1)

**Verification Skipped**

The verification should be similar to the Unif example. 

## RevBin 
Proposed invariant **I** is

	wp(RevBin) = [x > 0] * (z + x*(1/p)) + [x <= 0](z)

**Verified:**

	wp(body, I)
	= p * ([x-1 > 0] * (z + 1 + (x-1)*(1/p)) + [x - 1 <= 0](z + 1)) 
	+ (1-p) * ([x > 0] * (z + 1 + x*(1/p)) + [x <= 0](z + 1)) 
	= [x <= 0](z + 1) + [x = 1](z + 1 + x *(1-p)/p) + [x > 1]*(z + 1 + x*(1/p) - p/p)
	
Then

	[G]*wp(body, I)
	= [x > 0] ([x = 1](z + 1 + (1-p)/p) + [x > 1]*(z + x*(1/p)))
	= [x > 0] ([x = 1](z + 1/p) + [x > 1]*(z + x*(1/p)))
	= [x > 0] (z + x*(1/p))

Thus, `[G]*wp(body, I) + [not G]*z = I`. 