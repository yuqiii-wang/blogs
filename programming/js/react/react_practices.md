# React Hooks and Re

## Object, Function and Class

|Function|Object|Class|
|-|-|-|
|`function name() { ... }` or `const fn = () => {}`|`{ key: value, ... }`|`class Counter extends React.Component {...}`|

### Functional vs Class Component

* Class Component

It must include a `render()` method that returns JSX.

```js
class MyComponent extends React.Component {
  render() {
    return <h1>Hello, {this.props.name}!</h1>;
  }
}
```

* Functional Component

No `render()` method.

Lightweight and more concise compared to class components.

```js
function MyComponent(props) {
  return <h1>Hello, {props.name}!</h1>;
}
```

|Function Component|Class Component|
|-|-|-|
|State Mgt|Uses hooks like `useState`|Uses `this.state` and `this.setState()`|
|Lifecycle Methods|Uses hooks like `useEffect` for side effects.|Uses lifecycle methods like `componentDidMount`.|
|Syntax|Simple and Concise|Requires `constructor`, `this`, and `render()`.|
|Performance|Slightly more performant as they avoid class instantiation overhead.|Heavier runtime due to class instantiation.|

## Common ESLint Warnings and Best Coding Practices

### Using Template Literals

Prefer template literals over string concatenation.

Bad example:

```js
badStr = var1 + " " + var2;
```

Good example:

```js
goodStr = `${var1} ${var2}`;
```
