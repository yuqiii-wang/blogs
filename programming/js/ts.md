# TypeScript & JavaScript (ES Standards)

## Overview

**TypeScript** is a superset of JavaScript that adds static type checking and advanced features. All valid JavaScript is valid TypeScript. TypeScript extends JavaScript with:

* Type annotations and inference
* Interfaces and type aliases
* Enums and discriminated unions
* Generics for reusable components
* Access modifiers (public, private, protected)
* Decorators
* Abstract classes and advanced OOP

TypeScript is often called "Typed JavaScript" and compiles to JavaScript for execution in browsers and Node.js.

## JS and TS Unique Syntax

These are some JS and TS Unique Syntax rules that would be alien to typical python/java/c++ programmers.

### Arrow Functions: Implicit Return

No `return` keyword needed when the body is a single expression (no braces):

```js
const double = x => x * 2;         // No braces = implicit return
const add = (a, b) => a + b;
const getObj = (x) => ({ value: x }); // Parentheses needed to return an object literal

// With braces, return must be explicit
const addExplicit = (a, b) => { return a + b; };
```

### Destructuring: `:` Means Rename, Not Type

The `:` in object destructuring renames the variable — it does **not** mean type:

```js
const user = { name: "Bob", age: 30 };

const { name: userName, age: userAge } = user; // userName = "Bob", userAge = 30
// NOT: const { name: string } — that would make a variable called "string"

// Skip array elements with empty commas
const [first, , third] = [1, 2, 3]; // first=1, third=3

// Rest: pack remaining into new variable
const { name, ...rest } = user;         // rest = { age: 30 }
const [head, ...tail] = [1, 2, 3, 4];  // head=1, tail=[2,3,4]
```

### `??` vs `||`: Nullish Coalescing

`||` triggers on any falsy value (`0`, `""`, `false`). `??` only triggers on `null`/`undefined`:

```js
const volume = 0;
volume || 50;  // → 50  (wrong! 0 is falsy)
volume ?? 50;  // → 0   (correct: 0 is a valid value)

// Logical assignment (ES2021)
user.role ??= 'guest'; // Assign only if null/undefined
```

### Optional Chaining `?.`

Returns `undefined` instead of throwing on null access — no equivalent in Java/C++:

```js
const zip = user?.address?.zip;   // undefined, not TypeError
const id  = items?.[10]?.id;      // undefined if index out of bounds
const val = obj?.method?.();      // calls method only if it exists
```

### Spread `...` in Objects

Spread an object's properties into another (no equivalent in Java):

```js
const base = { host: 'localhost', port: 3000 };
const override = { ...base, port: 8080 }; // { host: 'localhost', port: 8080 }
```

### Named vs Default Module Exports

A file can export one `default` and many named exports — they are imported differently:

```js
// math.js
export const add = (a, b) => a + b;     // named
export default function multiply(...) {} // default

// importer.js
import multiply, { add } from './math.js'; // default has no braces, named does
import * as math from './math.js';         // import all named as namespace
const m = await import('./math.js');       // dynamic import (ES2020)
```

### `undefined` vs `null`

Two distinct "nothing" values — `undefined` is assigned automatically by the engine, `null` by the user:

```js
typeof undefined; // "undefined"
typeof null;      // "object"  ← famous quirk, not a real object

let x;      // x is undefined (declared but not assigned)
let y = null; // y is null (explicitly empty)
```

### `var` Hoisting (Avoid; Use `let`/`const`)

`var` is function-scoped and hoisted to the top of its function — it leaks out of blocks:

```js
for (var i = 0; i < 3; i++) {}
console.log(i); // 3 — i leaked! (use let to avoid this)
```

## TypeScript: Alien Syntax

### Structural Typing (Duck Typing)

Unlike Java/C++ where types are **nominal** (two classes with the same fields are still different types), TypeScript uses **structural typing** — if it has the right shape, it passes:

```ts
interface Point { x: number; y: number; }

function print(p: Point) { console.log(p.x, p.y); }

const obj = { x: 1, y: 2, z: 3 }; // Extra field z is fine
print(obj); // ✓ — Java would reject this
```

### Union Types and Literal Types

A variable can be typed as **one of multiple types** — there's no equivalent in Java/C++:

```ts
let id: string | number;  // can be either
id = 123;   // ✓
id = "abc"; // ✓

// Literal type: value IS the type
let dir: "left" | "right" | "up" | "down";
dir = "left"; // ✓
// dir = "diagonal"; // ✗ error
```

### Optional `?` and `readonly` in Interfaces

```ts
interface User {
  id: number;
  name: string;
  email?: string;           // Optional — may be absent entirely (not null)
  readonly createdAt: Date; // Cannot be reassigned after creation
}
```

`?` on an interface property means the key may not exist at all — different from `null`.

### Constructor Parameter Shorthand

TypeScript can declare **and** initialize class fields directly in the constructor parameters:

```ts
// ❌ Verbose (Java-style)
class User {
  private name: string;
  public age: number;
  constructor(name: string, age: number) {
    this.name = name;
    this.age = age;
  }
}

// ✓ TypeScript shorthand (identical result)
class User {
  constructor(private name: string, public age: number) {}
}
```

### Utility Types: `Partial`, `Pick`, `Omit`, `Record`

Built-in type transformations — no equivalent in Java/C++:

```ts
interface User { id: number; name: string; email: string; }

type PartialUser  = Partial<User>;            // All fields optional
type UserPreview  = Pick<User, "id" | "name">; // Only id and name
type UserNoId     = Omit<User, "id">;          // Everything except id
type Permissions  = Record<"read" | "write", boolean>;
// → { read: boolean; write: boolean }
```

### `keyof` and Indexed Access Types

```ts
interface User { id: number; name: string; }

type UserKey    = keyof User;      // "id" | "name"
type IdType     = User["id"];      // number  ← index types like array indexing but on types
type ValueTypes = User[keyof User]; // number | string
```

### Declaration Files (`.d.ts`)

A `.d.ts` file is a **pure type manifest** — no runtime code, only type signatures. Used to describe the types of a JavaScript library that has no TypeScript source:

```ts
// lodash.d.ts  (ships alongside lodash's .js files)
export function chunk<T>(array: T[], size: number): T[][];
export function flatten<T>(array: T[][]): T[];
```

TypeScript reads `.d.ts` at compile time for type checking, then ignores it at runtime. Think of it like a `.h` header in C++ but purely for types.

### Conditional Types

Types can have `if/else` logic — completely novel to Java/C++:

```ts
type IsArray<T> = T extends any[] ? true : false;
type A = IsArray<string[]>; // true
type B = IsArray<number>;   // false

// Extract the element type from an array type
type ElementOf<T> = T extends (infer U)[] ? U : never;
type E = ElementOf<string[]>; // string
```

## TypeScript to JavaScript Compilation



### `tsc` and `tsconfig.json`

A `tsconfig.json` file (placed at **project root**) tells the TypeScript compiler (`tsc`) how to compile `.ts` files.

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020", "DOM"],
    "outDir": "./out",
    "rootDir": "./src",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "moduleResolution": "node",
    "noImplicitAny": false,
    "strictNullChecks": false,
    "strictFunctionTypes": false,
    "noUnusedLocals": false,
    "noUnusedParameters": false,
    "noImplicitReturns": false,
    "allowSyntheticDefaultImports": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "out", "test", "**/*.test.ts"],
  "ts-node": {
    "compilerOptions": {
      "module": "commonjs"
    }
  }
}
```

where

* `ES2020` syntax but uses CommonJS modules (not ES6 `import`/`export`).
* `lib: ["ES2020", "DOM"]` — provides type definitions for `ES2020` built-ins (`Promise`, `Map`, etc.) and browser DOM (`document`, `window`, etc.). This allows using both Node.js and browser APIs.
* `"sourceMap": true`: Creates source maps so debuggers can map compiled `.js` back to original `.ts` source code.
    * Useful for debugging in Node.js or browser dev tools.
* `"declaration": true`: Generate `.d.ts` files
* `"include"`/`"exclude"`
    * `"include"`: Only compile files in `src/`
    * `"exclude"`: Skip `node_modules/`, `out/`, `test/`, and any `*.test.ts` files
* `"strict": true` combined with individual strict checks disabled to loosely validate types
    * Despite enabling `strict` mode, individual options explicitly disable checks
    * `noImplicitAny: false`: Accept untyped parameters/allows implicit `any` types
    * `strictNullChecks: false` allows unsafe `.length` on potentially null values
    * `strictFunctionTypes: false`: Accept incompatible signatures
    * `noImplicitReturns: false`: Allow missing returns


