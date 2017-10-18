---
layout: post
title:  Interview Cheatsheet
description: ongoing short question review for phone interviews
date:   2017-10-18 05:40:00
comments: true
---

## Introduction
As I go through the interview process, :pensive:, I thought it would be useful to keep a list of phone interview questions and their answers handy.  Thus far, there seems to be a lot of overlap between different companies' questions, which is an unexpected plus.

### Q1: What is an abstract class?  How does it relate to inheritance and interfaces?
A: An abstract class is a collection of base properties to define a certain class of objects.  I've also seen it called a partially incomplete object.  I don't necessarily mean object in the OO sense here.  A good example would be to consider an elevator.  One property of elevators that is important in their operation is how many people can ride on them at any given time.  So ```Elevator``` would be the abstract class, with virtual method ```numPassengers```.  Specific elevators will have different capacities, and will therefore need to implement the class-specific values and methods which identify it.  Some methods are universal to the ```Elevator``` class though, like what floor it's at, whether it's currently going up or down.  So, these methods could be implemented in the base class and inherited by children.  To summarize, an abstract class is partially incomplete.

In contrast, an interface is totally incomplete.  It basically just sets up an object with methods that are purely virtual.  An example would be a car class.  You could set up an interface something like this 

```c++
interface Car:
    public:
        virtual void start();
```

From this, you could create new objects that inherit from ```Car```:

```c++
Car c1 = new Car(){
	void start(){
        // start procedure here ...
    }
}
```

The purely virtual method ```start()``` is explicitly defined within the definition of ```c1```.

There is a pretty good discussion [here](https://stackoverflow.com/questions/761194/interface-vs-abstract-class-general-oo).

### Q2:  What are reflection and introspection?
A:  These are two words for the same language-adjacent feature.  A reflection- and/or introspection-oriented program can modify its behavior at runtime to achieve desired goals (e.g. runtime, memory-allocation, etc...).  According to [this](http://jackieokay.com/2017/04/13/reflection1.html), the desire to use such methods within C++ is there due to potential impacts when scaling up existing programs.  (Purely virtual methods and parent-child inheritance paradigms can often result in a performance hit.)

### Q3: What is templating?
Templating is a generic way of defining classes that may have different types despite each implementation having the same method definitions, e.g. a matrix class could be templated to do linear algebraic manipulation using instances created using any type.  Potential drawbacks would be suboptimal execution due to overgeneralization.

### Q4: Thread management:  What is deadlock?  What are race conditions?
Deadlock:
Resources A and B are each being used by two threads X and Y
* X starts using A (locking it)
* X and Y start using B
* Y gets priority from the OS.  X still has a lock on A.
* Y needs to use A, but it's locked
* X is waiting for Y to finish executing B

This is known as deadlock, when two competing resources freeze program execution due to improper thread management.  One way to combat this is to avoid excessive locks, or through the use of critical sections.

Race conditions:
Behavior of software or hardware components where system output depends on the sequencing of uncontrollable events, like how the OS manages running threads.

### Q5:  Pointer vs. reference
* Pointers can point nowhere, as in ```int *p = NULL```. 
* References are static and can't be changed after binding.
* You can traverse data structures by using pointer arithmetic, which you cannot due with references.  You can however take an object pointed to by a reference and do arithmetic on that, something like ```int *p = &obj + 5```.  See [this](https://stackoverflow.com/questions/57483/what-are-the-differences-between-a-pointer-variable-and-a-reference-variable-in).