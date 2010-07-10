% Evaluating math with Haskell
% Aur Heru Saraf

Ignore this
===========

> {-# LANGUAGE GeneralizedNewtypeDeriving #-}
> module Presentation where
> import qualified List
> import Data.List
> import Data.Ord
> import Monad

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
> move (Player _ f) = f
>
> data Players = Players Player Player
> 
> data Score = Lose | NotOver | Win deriving (Eq, Ord)
> 
> data Nim = Nim [Matches]
>     deriving Show
> 
> newtype Matches = Matches Int deriving (Ord, Eq, Enum, Num)
> instance Show Matches where show (Matches k) = show k ++ " match(es)"
> 
> data Move = Move Int Matches
> instance Show Move where
>   show (Move i (Matches k)) = "Remove " ++ show k ++
>                               " from stack " ++ show (i + 1)

Rules of the game
=================

> apply :: Move -> Nim -> Nim
> apply (Move i k) (Nim stacks) | k > 0 && s >= k = Nim (h ++ [s-k] ++ t)
>   where (h, (s: t)) = splitAt i stacks
> apply _ _ = undefined

> score :: Nim -> Score
> score (Nim stacks) | all (== 0) stacks = Lose
> score _ = NotOver

Main loop
=========

> playNim :: Players -> Nim -> IO Player
> playNim (Players p1 p2) nim = do
>     putStrLn (show p1 ++ "'s turn")
>     m <- move p1 nim
>     putStrLn (show p1 ++ "'s move is: " ++ show m)
>     let nim' = apply m nim
>     let result = if score nim' == Lose 
>                  then return p2 
>                  else playNim (Players p2 p1) nim'
>     result
>

return :: (Monad m) => a -> m a
instance Monad IO

Human Player
============

> humanMove :: Nim -> IO Move
> humanMove nim = do
>     putStrLn (showNim nim)
>     putStrLn ("Choose a stack:")
>     stack <- getLine
>     putStrLn ("Choose amount:")
>     amount <- getLine
>     return (Move (read stack - 1) (Matches (read amount)))
> 
> showNim :: Nim -> String
> showNim (Nim stacks) =
>   unlines [show i ++ ": " ++ show k | (i, k) <- zip [1..] stacks]
>
> humanPlayer name = Player name humanMove
> 
> humanVsHumanNim nim = playNim players nim 
>   where players = Players (humanPlayer "Aur") (humanPlayer "Oleg")

Runtime bugs I had
==================

When I first ran the code, I found the following bugs:

- Confused i and k in Move's show

- Confused i and k in apply

(i and k were both Ints). So I added the newtype Matches.

- The condition for apply was s > k and not s >= k

- I made Matches a Read instance, so it tried to parse from user input
a string like "Matches 1" (solution: read as Int and construct Matches
myself)

And then it worked!
===================

- How many bugs would you have had for a program this size that you
run for the first time in C? In Python?

- I usually have a very high bug count for a programmer

- All of these bugs were immediately apparent. No edge case chasing.

Computing a move's result
=========================

> evaluateAllMoves :: Nim -> [(Move, Score)]
> evaluateAllMoves nim = map evaluate (legalMoves nim)
>   where evaluate m = (m, score' (apply m nim))
> 
> bestMove nim = maximumBy (comparing snd) (evaluateAllMoves nim)
>
> score' nim = case score nim of
>   NotOver -> reverseScore (snd (bestMove nim))
>   s -> s
> 
> legalMoves :: Nim -> [Move]
> legalMoves (Nim stacks) = [Move i k | i <- [0..length stacks - 1],
>                                             k <- [1..stacks !! i]]
> 
> reverseScore :: Score -> Score
> reverseScore Lose = Win
> reverseScore Win = Lose
> reverseScore NotOver = NotOver
>
> computerMove nim = return (fst (bestMove nim))
> 
> humanVsComputerNim nim = playNim players nim
>   where players = Players (humanPlayer "Aur") 
>                           (Player "Oleg" computerMove)

First run
=========

Seemed to work! I lost every time I was the losing player or made a mistake!

Exploring the game
==================

First, lets write a pure solver

> data PurePlayer = PurePlayer String (Nim -> Move)
> instance Show PurePlayer where show (PurePlayer name _) = name
> pureGetMove (PurePlayer _ f) = f
> data PurePlayers = PurePlayers PurePlayer PurePlayer
> pureMove nim = fst (bestMove nim)
> computerPlayer name = PurePlayer name pureMove
> pureNim (PurePlayers _ p2) nim | score nim == Lose = p2
> pureNim (PurePlayers p1 p2) nim =
>   let m = pureGetMove p1 nim
>       nim' = apply m nim
>   in if score nim' == Lose 
>      then p2 
>      else pureNim (PurePlayers p2 p1) nim'
>
> computerNim nim = pureNim players nim
>   where players = PurePlayers (computerPlayer "A") (computerPlayer "B")
>
> nim2WinTable n =[[computerNim (Nim [i, j]) | j <- range] | i <- range]
>   where range = [0..n]

nim2WinTable 4 gives us:

[[B,B,A,A,A],
 [B,A,A,A,A],
 [A,A,B,A,A],
 [A,A,A,B,A],
 [A,A,A,A,B]]