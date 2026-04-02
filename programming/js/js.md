# JavaScript

## DOM Element Manipulation

JavaScript is designed to manipulate DOM (Document Object Model) elements to control UI display.

DOM is used to render a UI page, where compiler, e.g., chrome V8 engine, parses a DOM to a tree-structured elements and display such elements per style.
User can interact with DOM elements.

For example, this DOM

```html
<!DOCTYPE html>
<html>
<head>
    <title>My Web Page</title>
</head>
<body>
    <h1>Welcome to My Web Page</h1>
    <p>This is a simple paragraph.</p>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
        <li>Item 3</li>
    </ul>
</body>
</html>
```

is parsed to

```txt
Document
├── html
│   ├── head
│   │   └── title ("My Web Page")
│   └── body
│       ├── h1 ("Welcome to My Web Page")
│       ├── p ("This is a simple paragraph.")
│       └── ul
│           ├── li ("Item 1")
│           ├── li ("Item 2")
│           └── li ("Item 3")
```

For any element change, traditionally, the whole UI page needs re-rendering.
React is used to prevent full webpage change by taking DOM partial update only.

React relies on a virtual DOM, which is a copy of the actual DOM.
React's virtual DOM is immediately reloaded to reflect this new change whenever there is a change in the data state.
After which, React compares the virtual DOM to the actual DOM to figure out what exactly has changed.

### Common DOM Elements

|Element|Display Type|Primary Use Case & Key Differences|
|:---|:---|:---|
|`<div>`|Block|Generic use for block|
|`<span>`|Inline|Used to hook onto a piece of text for styling (e.g., changing color, adding an icon) without affecting layout.|
|`<p>`|Block|For a paragraph of text. It is not for grouping other elements.|
|`<ul>`, `<ol>`|Block|Containers for unordered (bulleted) and ordered (numbered) lists, respectively. Their only valid direct child is the `<li>` element.|
|`<li>`|List Item|It must be a child of a `<ul>` or `<ol>`. Behaves mostly like a block element but has a list marker (bullet/number).|

where for `display: block;` vs `display: inline;`, the `inline` display aims to fit content within only necessary space, e.g., ignores width and height.

## JavaScript ES Standards Timeline

JavaScript ES Standards (also called ECMAScript) are official specifications for JavaScript. "ES" stands for ECMAScript Standard.

By 2026, the ES developments are as below.

| Standard | Release | Key Features |
|:---|:---|:---|
| ES5 (ECMAScript 5) | 2009 | `strict mode`, `Object.create()`, getters/setters, `Array.prototype` methods |
| ES6 / ES2015 | 2015 | `class`, `let`/`const`, arrow functions, template literals, destructuring, `Promise`, modules |
| ES2016 | 2016 | `Array.prototype.includes()`, exponentiation operator (`**`) |
| ES2017 | 2017 | `async`/`await`, `Object.entries()`, `Object.values()` |
| ES2018 | 2018 | Rest/spread in objects, `Promise.prototype.finally()`, async iterators |
| ES2019 | 2019 | `Array.prototype.flat()`, `Array.prototype.flatMap()`, `Object.fromEntries()` |
| ES2020 | 2020 | Optional chaining (`?.`), nullish coalescing (`??`), `BigInt`, dynamic `import()` |
| ES2021+ | 2021+ | Logical assignment (`&&=`, `\|\|=`), numeric separators, `Promise.any()` |

## Single thread and Event Loop

* JS is single-thread

JS is an asynchronous and single-threaded interpreted language. A single-thread language is one with a single call stack and a single memory heap.

* Event Loop

Call Stack: If your statement is asynchronous, such as `setTimeout()`, `ajax()`, `promise`, or click event, the event is pushed to a queue awaiting execution.

Queue, message queue, and event queue are referring to the same construct (event loop queue). This construct has the callbacks which are fired in the event loop.

Modern JS uses features like `Promises` and the `async`/`await` syntax to give better performance.

### Parallelism by Web Workers

Web Workers can give multiple threads for execution.

## JS Syntax

### `this` Binding: Changes With Call Context

Unlike Java/C++ where `this` always refers to the current instance, in JS `this` depends on **how the function is called**, not where it's defined:

```js
const obj = {
  name: "Alice",
  greetRegular: function() { console.log(this.name); }, // this = obj ✓
  greetArrow:   () =>        { console.log(this.name); }, // this = outer scope ✗
};
obj.greetRegular(); // "Alice"
obj.greetArrow();   // undefined — arrow functions do NOT bind their own `this`

// Passing a method as callback loses `this`
setTimeout(obj.greetRegular, 100); // undefined — `this` is now window/undefined
setTimeout(() => obj.greetRegular(), 100); // "Alice" — arrow wrapper preserves context
```

**Rule:** Use regular functions for object methods; use arrow functions for callbacks inside them.

### Truthy / Falsy: Six Falsy Values

JS coerces any value to boolean in `if` / `||` / `&&`. Only **six values are falsy** — everything else is truthy:

```js
// Falsy: false, 0, "" (empty string), null, undefined, NaN
if (0)         {} // skipped
if ("")        {} // skipped
if (null)      {} // skipped

// Truthy surprises (these all pass)
if ([])        {} // empty array is truthy!
if ({})        {} // empty object is truthy!
if ("0")       {} // non-empty string is truthy!
```

### `==` vs `===`: Type Coercion vs Strict Equality

`==` coerces types before comparing — results are unintuitive. Always use `===`:

```js
0   == "0"    // true  ← string coerced to number
0   == false  // true  ← false coerced to 0
""  == false  // true
null == undefined // true  ← special rule

0   === "0"   // false ✓
null === undefined // false ✓
```

### `+` Operator: Addition or Concatenation

`+` with any string coerces to string concatenation, not addition:

```js
1 + 2 + "3"  // "33"  (left-to-right: 3, then "3" + "3")
"1" + 2 + 3  // "123"
"5" - 1      // 4     (- only does numeric, coerces "5" to 5)
```

### Closures: Functions Capture Surrounding Scope

A function remembers the variables from the scope it was **defined** in, even after that scope has exited — no analog in Java/C++:

```js
function makeCounter() {
  let count = 0;                    // `count` lives inside makeCounter
  return () => { count++; return count; }; // inner function captures `count`
}

const counter = makeCounter();      // makeCounter() has finished executing...
counter(); // 1 — ...but count is still alive, captured by the closure
counter(); // 2
counter(); // 3
```

Classic bug — `var` in a loop closures over the same variable:

```js
for (var i = 0; i < 3; i++) {
  setTimeout(() => console.log(i), 0); // prints 3, 3, 3 — all closures share one `i`
}
for (let i = 0; i < 3; i++) {
  setTimeout(() => console.log(i), 0); // prints 0, 1, 2 — `let` creates a new binding per iteration
}
```

### JS Prototype Inheritance

JS is a scripting language not required compilation such as observed in c++ and java, as a result, the JS hidden property `Prototype` much more flexible in inheritance against c++ or java parent class.

For example, `Person.prototype.greet` can be added just in time without prior declaration that is often observed in c++ and java.

```js
// 1. Define a constructor function
function Person(name, age) {
  this.name = name;
  this.age = age;
}

// 2. Add a method to the Person function's prototype object
Person.prototype.greet = function() {
  console.log(`Hello, my name is ${this.name}.`);
};

// 3. Let's inspect the prototype itself
console.log(Person.prototype);

// greet: f(): The function we just defined.
// constructor: f Person(name, age): A property that points back to the Person function itself. This is a default property on every function's prototype.
// __proto__: Object: This shows that the Person.prototype object itself has a prototype, which is the base Object.prototype. This is the foundation of the prototype chain.

// 4. Create an instance of Person
const person1 = new Person('Alice', 30);

// 5. Inspect the instance
console.log(person1);

// name: "Alice"
// age: 30
// __proto__: Object: This is the crucial link. It points to the object that serves as the prototype for person1.
```

Now create inheritance `Student`.

```javascript
function Student(name, age, studentId) {
  Person.call(this, name, age);
  this.studentId = studentId;
}

Student.prototype = Object.create(Person.prototype);
Student.prototype.constructor = Student;

// Add a method specific to Student
Student.prototype.study = function() {
  console.log(`${this.name} is studying.`);
};

const student1 = new Student('Bob', 22, 'S12345');

student1.greet(); // Output: Hello, my name is Bob. (Inherited!)
student1.study();    // Output: Bob is studying. (Own prototype method)

console.log('\n--- Inspecting the prototype chain ---');
console.log('Student instance:', student1);
console.log('Student.prototype:', Student.prototype);
console.log('Person.prototype:', Person.prototype);
```

#### Prototype Chain

To access a property on an object, JavaScript's engine will

1. first look for the property on the object itself
2. if it doesn't find it, it will then look at the Object's prototype
3. continues up what is known as the prototype chain until `Object`
4. if still not found the end of the chain is reached (which is `null`)

```js
// This demonstrates the chain:
// student1 -> Student.prototype -> Person.prototype -> Object.prototype -> null
console.log(Object.getPrototypeOf(student1) === Student.prototype); // true
console.log(Object.getPrototypeOf(Student.prototype) === Person.prototype); // true
```

#### Common Confusion: `__proto__` vs `prototype`

* `__proto__` serves as pointer for fast object location
* `prototype` is the actual object

To execute `student1.study();`

1. JS checks `student1` for a study method. Not found.
2. JS follows `student1.__proto__` to `Student.prototype`.
3. JS checks `Student.prototype` for a study method. Found! It executes the function.

To execute `student1.greet();`

1. JS checks `student1` for a greet method. Not found.
2. JS follows `student1.__proto__` to `Student.prototype`.
3. JS checks `Student.prototype` for a greet method. Not found.
4. JS follows `Student.prototype.__proto__` to `Person.prototype`.
5. JS checks `Person.prototype` for a greet method. Found! It executes the function.

## Execution Env: Node JS vs Browser Chrome

* Browser: The browser provides a **client-side execution environment**

Designed to create interactive and dynamic web pages, manipulate the Document Object Model (DOM), and respond to user events like clicks and keyboard presses, and communicate with servers.

JavaScript runs in a browser-restricted sandboxed environment with limited access to computer resources, e.g., filesystem, OS executables.

* Node.js: Node.js offers a **server-side runtime environment**

Allow developers to use JavaScript to build back-end services, APIs, command-line tools, and other applications that run outside of a browser.

Node.js does not have a DOM because it doesn't render HTML pages.
Consequently, user cannot access objects like `document` or `window` in a Node.js environment.

### Node.js Backend vs Frontend Differences

||Node.js Backend|Browser/Frontend|
|:---|:---|:---|
|Module System|CommonJS (`require`, `module.exports`)|ES modules (`import`, `export`) or bundled CommonJS|
|File System|`fs` module|No file system access|
|Network|`http`, `https`, `net` modules|`fetch`, `WebSocket`|
|Process|process object, environment variables|No process access|
|Globals|`global`, `__dirname`, `__filename`|`window`, `document`, `navigator`|
|Threading|Worker threads|Web Workers (limited)|
|OS Access|Full OS access (files, processes)|Sandboxed, very limited|
|Package Manager|`npm`, `yarn`|Bundling tools needed, e.g., `Webpack`, `Vite`|
---

where

* Node.js Backend (`http`, `https`, `net`) can create server
* Browser Frontend (`fetch`, `WebSocket`) Can only request, not create servers; Sandboxed by browser (CORS restrictions)

## JS, TS and TSX Grammar Comparison

|Feature|`.js`|`.ts`|`.tsx`|
|:---|:---|:---|:---|
|Language|JavaScript|TypeScript|TypeScript|
|JSX Support|Yes (with a tool like Babel)|No|Yes|
|Type Safety|No|Yes|Yes|
|Common Use|Basic React components|Non-component TypeScript logic|React components in TypeScript|
