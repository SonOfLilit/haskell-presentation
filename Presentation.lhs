% Evaluating math with Haskell
% Aur Heru Saraf

This presentation is Literate Haskell
=====================================

This means it can be executed.

Indeed, if you are reading it at home, feel free to load it in GHCi
and play with the definitions.

So lets start with some imports, which you can ignore...

> {-# LANGUAGE GeneralizedNewtypeDeriving #-}
> module Presentation where
> import qualified List
> import Data.List
> import Data.Ord

Hello, my name is Haskell
=========================

Pure functional, lazy, type inferring programming language
----------------------------------------------------------

Good things people say about me
===============================

- WOW! I basically wrote this without testing just thinking about my
program in terms of transformations between types. I wrote the
test/example code and had almost no implementation errors in the code!
The compiler/type-system is really really good at preventing you from
making coding mistakes! I've never in my life had a block of code this
big work on the first try. I am WAY impressed.

Good things people say about me
===============================

- I learned Haskell a couple of years ago, having previously
programmed in Python and (many) other languages. Recently, I've been
using Python for a project (the choice being determined by both
technical and non-technical issues), and find my Python programming
style is now heavily influenced (for the better, I hope ;-) by my
Haskell programming experience.

Bad things people say about me
==============================

- As with pretty much any given functional programming language,
things are so unintuitive that, well, I can't even explain properly
how BAD things are. I'm studying haskell for college and I must say,
when I compare the things I can do with C (I can even use pointers
well), Java (I solve problems with classes), Python (I solve problems
in simpler ways than C or Java) and even Shell Scripts (and that's
something!) with Haskell (I pretty much can't do a thing), I know
something must have gone terribly wrong!  

Reading Haskell
===============

Don't panic!

Definitions, Type Declarations, IO
==================================

> main :: IO ()
> main = putStrLn "Hello, World!"

Functions, Lists, Pattern Matching
==================================

> head' :: [a] -> a
> head' (x:_) = x
> head' [] = undefined

undefined :: a

Indentation, Currying, ...
==========================

... Higher-Order Functions, Lazyness, Type Inference
----------------------------------------------------

> fib n = fib' !! n
>   where fib' = (0 : 1 : zipWith (+) fib' (tail fib'))

(!!) :: [a] -> Int -> a

zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]

Algebraic Data Types
====================

> data Bool' = False' | True'
>
> not' :: Bool' -> Bool'
> not' False' = True'
> not' True' = False'

> data List a = Empty | a :. List a
> 
> tail' :: List a -> List a
> tail' (_ :. xs) = xs
> tail' Empty = undefined

Type Classes
============

> maxSucc :: (Ord a) => [a] -> a
> maxSucc a = maximum (List.delete (maximum a) a)

Writing Type Classes
====================

> data Peano = Zero
>            | Succ Peano
>    deriving (Show, Read, Eq)
> 
> instance Ord Peano where
>     compare (Succ x) (Succ y) = compare x y
>     compare Zero     (Succ _) = LT
>     compare (Succ _) Zero     = GT
>     compare Zero     Zero     = EQ
> 
>     min (Succ x) (Succ y) = Succ (min x y)
>     min _        _        = Zero

...
===

>     max (Succ x) (Succ y) = Succ (max x y)
>     max Zero     y        = y
>     max x        Zero     = x
> 
>     _      < Zero   = False
>     Zero   < _      = True
>     Succ n < Succ m = n < m
> 
>     x > y  = y < x
> 
>     x <= y = not (y < x)
> 
>     x >= y = not (x < y)

(From <http://code.haskell.org/~thielema/htam/src/Number/PeanoNumber.hs>)

Let's play a game
=================

We all know Nim.

- n stacks of matches on a table

- Each player, in his turn, must remove one or more matches from a
single stack

- If a player cannot, he loses

Data Structure
==============

> data Player = Player String (Nim -> IO Move)
> instance Show Player where show (Player name _) = name
> getMove (Player _ f) = f
> 
> data Score = Lose | NotFinished | Win deriving (Eq, Ord)
> 
> data Nim = Nim [Matches]
>     deriving Show
> newtype Matches = Matches Int deriving (Ord, Eq, Enum, Num)
> instance Show Matches where
>   show (Matches k) = show k ++ " match(es)"
> 
> data Move = Move Int Matches
> instance Show Move where
>   show (Move i (Matches k)) = "Remove " ++ show k ++
>                               " from stack " ++ show (i + 1)

Rules of the game
=================

> apply :: Move -> Nim -> Nim
> apply (Move i k) (Nim stacks) | legal = Nim (h ++ [s-k] ++ t)
>   where (h, (s: t)) = splitAt i stacks
>         legal = k > 0 && s >= k
> apply _ _ = undefined

> score :: Nim -> Score
> score (Nim stacks) | all (== 0) stacks = Lose
> score _ = NotFinished

Main loop
=========

> playNim :: Player -> Player -> Nim -> IO Player
> playNim p1 p2 nim = do
>     putStrLn (show p1 ++ "'s turn")
>     m <- getMove p1 nim
>     putStrLn (show p1 ++ "'s move is: " ++ show m)
>     let nim' = apply m nim
>     let result = if score nim' == Lose 
>                  then return p2 
>                  else playNim p2 p1 nim'
>     result

return :: (Monad m) => a -> m a
instance Monad IO

Human player
============

> humanMove :: Nim -> IO Move
> humanMove nim = do
>     putStrLn (showNim nim)
>     putStrLn "Choose a stack:"
>     stack <- getLine
>     putStrLn "Choose amount:"
>     amount <- getLine
>     return (Move (read stack - 1) (Matches (read amount)))
> 
> showNim (Nim stacks) =
>   unlines [show i ++": "++ show k | (i, k) <- zip [1..] stacks]
> 
> human name = Player name humanMove
> humanVsHumanNim = playNim (human "Aur") (human "Heru") 

Runtime bugs I had
==================

When I first ran the code, I found the following bugs:

- Confused i and k in Move's show

- Confused i and k in apply

(i and k were both Ints). So I added the newtype Matches.

- The condition for apply was s > k and not s >= k

- I derived Read for Matches, so it tried to parse from user input a
string like "Matches 1" (solution: read as Int and construct Matches
from it myself)

And then it worked!
===================

- How many bugs would you have had for a program this size that you
run for the first time in C? In Python?

- I usually have a very high bug count for a programmer

- All of these bugs were immediately apparent. No edge case chasing

- I never had a bug again in this area of the code

Computing an optimal move
=========================

> computerMove nim = return (fst (bestMove nim))
>
> bestMove nim = maximumBy (comparing snd) (evaluateAllMoves nim)
> evaluateAllMoves nim = map evaluate (legalMoves nim)
>   where evaluate m = (m, score' (apply m nim))
>
> score' nim = case score nim of 
>   Lose -> Lose
>   _ -> negateScore (snd (bestMove nim))
> negateScore Win = Lose
> negateScore Lose = Win
> 
> legalMoves :: Nim -> [Move]
> legalMoves (Nim stacks) = 
>    [Move i k | i <- [0..length stacks - 1],
>                k <- [1..stacks !! i]]

Let's play!
===========

> humanVsComputerNim = playNim (human "Aur") 
>                              (Player "Oleg" computerMove)

First run
=========

Seemed to work! I lost every time I was the losing player or made a
mistake

Exploring the game
==================

All this IO is very uncomfortable.

First, lets write a pure solver.

> data PurePlayer = PurePlayer String (Nim -> Move)
> instance Show PurePlayer where show (PurePlayer name _) = name
> pureGetMove (PurePlayer _ f) = f

Pure Nim
========

> pureComputerMove nim = fst (bestMove nim)
> 
> pureNim _ p2 nim | score nim == Lose = p2
> pureNim p1 p2 nim =
>   if score nim' == Lose 
>   then p2 
>   else pureNim p2 p1 nim'
>     where m = pureGetMove p1 nim
>           nim' = apply m nim
>
> computer name = PurePlayer name pureComputerMove
> computerNim = pureNim (computer "A") (computer "B")

And lets explore!
=================

> nim2Table n =
>   [[computerNim (Nim [i, j]) | j <- range] | i <- range]
>     where range = [0..n]

nim2Table 4 gives us:

    [[B,B,A,A,A],
     [B,A,A,A,A],
     [A,A,B,A,A],
     [A,A,A,B,A],
     [A,A,A,A,B]]

But it is so slow...
====================

nim2Table 7 takes 3.42 seconds on my old laptop, but nim2Table 8 takes
30 seconds, and it gets worse. Lets try to optimize a bit.

We will try to memoize results of bestMove.

To keep the code simple, we will only tackle nim2.

Memoizing
=========

> fastBestMove' (Nim [Matches a, Matches b]) = bests !! a !! b
> fastBestMove' nim = fastBestMove nim
> 
> bests = [[fastBestMove (Nim [a, b]) | b <- [0..]] | a <- [0..]]
> 
> fastBestMove nim = maximumBy (comparing snd)
>                              (fastEvaluateAllMoves nim)
> 
> fastEvaluateAllMoves nim = map evaluate (legalMoves nim)
>   where evaluate m = (m, fastScore' (apply m nim))
> 
> fastScore' nim = case score nim of 
>   Lose -> Lose
>   _ -> negateScore (snd (fastBestMove' nim))

And let's play!
===============

> fastComputerMove nim = fst (fastBestMove' nim)
> fastComputer name = PurePlayer name fastComputerMove
> 
> fastNim = pureNim (fastComputer "A") (fastComputer "B")
> 
> fastNim2Table n =
>   [[fastNim (Nim [i, j]) | j <- range] | i <- range]
>     where range = [0..n]

*Success!* fastNim2Table 20 executes within the blink of an eye
(profiler reports 0.04 seconds)

Parallelization
===============

What would it take to parallelize this?

Nothing! Just setting a compiler flag!

We only know how to do this for functional languages like Haskell.

Any Questions Up Until Now?
===========================

Don't worry, we are not done yet...

Parallelization, Abstraction, Mathematics
=========================================

I will show an example that I read at sigfpe's blog at
<blog.sigfpe.com/>, in a post titled "An Approach to Algorithm
Parallelization". He deserves the whole credit.

The problem
===========

Given a sequence of numbers that can be ordered, find the largest sum
that can be made from a subsequence of that sequence.

e.g.

    [1, 4, -6, 2, -2, 3, 5, 2, -1, 2, -5, 1, -10, 2]

The largest sum can be made from the subsequence

    [3, 5, 2, -1, 2]

A solution
==========

We can walk the sequence, keeping an accumulator that is reset to 0
every time it becomes negative, and remembering the largest value it
gets.

In Haskell:

> solution s = snd (solution' s (0, -infinity))
> solution' :: (Ord t, Num t) => [t] -> (t, t) -> (t, t)
> solution' [] (a, m) = (a, m)
> solution' (x:xs) (a, m) = solution' xs (a', m')
>   where a' = max (a + x) 0
>         m' = max a' m
>
> infinity :: Double
> infinity = 1/0

Testing it
==========

Todo: Introduce QuickCheck.

Running it
==========

It worked on first try.

    > solution [1, 4, -6, 2, -2, 3, 5, 2, -1, 2, -5, 1, -10, 2]
    11.0
    > solution [-1]
    0.0

Looks good.

Parallelizing it
================

This looks more problematic. We keep a running counter, and it is very
dependent on previous values.

    solution' [] (a, m) = (a, m)
    solution' (x:xs) (a, m) = solution' xs (a', m')
      where a' = max (a + x) 0
            m' = max a' m

The weird step
==============

Lets try to solve a simpler problem. We will replce max with addition
and addition with multiplication.

0 is the identity for addition so it should be replaced with 1, the identity for multiplication.

-infinity should similarly be replaced with 0.

> solution2 s = m
>   where (_, m) = solution2' s (1, 0)
> solution2' [] (a, m) = (a, m)
> solution2' (x:xs) (a, m) = solution2' xs (a', m')
>   where a' = (a * x) + 1
>         m' = a' + m

Why did we do this?
===================

Because solution2' now looks very close to being linear, and linear
functions are easy to parallelize.

Lets ignore the fact that it solves the wrong problem for now.

That "1" was iterferring with the linearity we want. Lets replace it
with a parameter on which the function will be linear.

> solution3 s = m
>   where (_, m, _) = solution3' s (1, 0, 1)
> solution3' [] (a, m, i) = (a, m, i)
> solution3' (x:xs) (a, m, i) = solution3' xs (a', m', i)
>   where a' = (a * x) + i
>         m' = a' + m

Linear functions
================

What can we do now? We can calculate (solution3' sequence) on any
vector by way of how it acts on base vectors:

    let f be solution3' sequence, then
    f (a, m, i) = a * f (0, 0, 1)  + m * f (0, 1, 0) + i * f (0, 0, 1)

The cool thing is, this *is* paralellizable.

Linear algebra
==============

Now we'll need some code for working with vectors, bases and matrices.

> type Vector t = (t, t, t)
> type Matrix t = Vector (Vector t)
> x, y, z :: (Num a) => Vector a
> x = (1,0,0)
> y = (0,1,0)
> z = (0,0,1)
> matrix :: (Num t) => (Vector t -> Vector t) -> Matrix t
> matrix f = (f x, f y, f z)
> (.+) :: (Num t) => Vector t -> Vector t -> Vector t
> (a, b, c) .+ (d, e, f) = (a+d, b+e, c+f)
> (.*) :: (Num t) => t -> Vector t -> Vector t
> a .* (b, c, d) = (a*b, a*c, a*d)
> (.*.) :: (Num t) => Matrix t -> Vector t -> Vector t
> (u, v, w) .*. (a, b, c) = (a .* u) .+ (b .* v) .+ (c .* w)

Applying
========

> solution4 s = m
>   where (_, m, _) = mat .*. (1, 0, 1)
>         mat = matrix (solution3' s)

But most of the work is done in calculating the matrix, before we even
get to (1, 0, 1)!

And we can parallelize it:

--## TODO: This line may be wrong. Do the math and correct
it. Preferrably after sleep.

    let f = solution3', then
    matrix (f (s1 ++ s2)) == (matrix f s2) * (matrix f s1)

