# Kotlin

## Kotlin Intro

It is a general-purpose language that runs on the Java Virtual Machine (JVM) and can also be compiled to JavaScript or native code.

Google declared Kotlin as its preferred language for Android app development since 2019.

## Kotlin Syntax

* `companion object`

Same as below, works like a static function without instantiating an object

```kotlin
class ToBeCalled {
    companion object {
        fun callMe() = println("You are calling me :)")
    }
}

fun main(args: Array<String>) {
    // Calling callMe() directly on the class, like a static method
    ToBeCalled.callMe() 
}
```