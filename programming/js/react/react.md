# React

## React Compilation

1. Transpiling (by Babel): Converting modern JavaScript and JSX into backward-compatible JavaScript that can run in a wider range of browsers.
2. Bundling (by webpack): Combining multiple JavaScript files and other assets (like CSS and images) into a smaller number of files to reduce the number of HTTP requests.
3. Optimization (by webpack): Minifying code, removing unused code ("tree shaking"), and other performance enhancements.

### Node JS

Node.js is a JavaScript runtime environment that runs JavaScript on the server-side (UI rendering and interaction is usually on the client side powered by engine, e.g., Chrome V8, but for development purpose there need hot-reloading, backend API simulation, etc.).

* Package Management: Node.js comes with `npm` (Node Package Manager)
* Running Build Tools: Babel and webpack are themselves Node.js modules. Node.js provides the environment to execute these tools and carry out the compilation and bundling process.
* Development Server: Tools like webpack-dev-server run on Node.js to provide a local development environment, e.g., hot-reloading

About `npm` and `npx`:

* `npm` (Node Package Manager) is the default package manager for Node.js.
* `npx` (Node Package Execute) is a tool that comes bundled with npm (since version 5.2.0) and is used to execute Node.js packages.

#### `npm` Instructions

* `npm init`: Initializes a new Node.js project by creating a `package.json` file; `npm init -y` helps quickly create a default package.json without going through the interactive prompts.

#### Production Dependencies vs. Development Dependencies

* dependencies: These are packages essential for application to run in a production environment, e.g., Express, React
* devDependencies: These are packages only necessary during the development process, e.g., testing (like Jest or Mocha), code bundling (like Webpack), and code quality (like ESLint and Prettier)

For prod:

* `npm install <pkg name>` or `npm install <pkg name> --save-prod`

For Dev

* `npm install <pkg name> --save-dev`

Some packages are command-line utilities used across multiple projects.

* `npm install -g <pkg name>`

#### `npm start` vs. `npm run`

In `package.json` file, the scripts object allows you to define custom commands to automate tasks.

`npm run <script-name>`: This is the standard way to execute any custom script defined in `package.json`.
For example, `npm run dev` will execute `nodemon src/index.js`.

```json
"scripts": {
 "start": "node build/index.js",
 "dev": "nodemon src/index.js",
 "test": "jest",
 "build": "tsc"
}
```

For `npm run start` there is a shortcut `npm start` (it is equivalent to executing `npm run start`).
If a "start" script is not defined, `npm start` will default to running `node server.js`.

### Babel

Babel is a toolchain that is mainly used to convert ECMAScript 2015+ code into a backwards compatible version of JavaScript in current and older browsers or environments.
In other words, transcript higher version JavaScript code into lower version JavaScript.

For example, arrow function is converted to JavaScript ES5 equivalent.

ES2015:

```js
// Babel Input: ES2015 arrow function
[1, 2, 3].map(n => n + 1);
```

ES5 equivalent:

```js
// Babel Output: ES5 equivalent
[1, 2, 3].map(function(n) {
  return n + 1;
});
```

### Webpack

In a React application, Webpack takes all individual JavaScript files, CSS, images, and other assets and bundles them together into a few optimized files.

1. Dependency Graph: Webpack starts from an entry point file (usually `index.js`) and builds a dependency graph of all the modules
2. Loaders:
  * babel-loader: This loader tells webpack to run all `.js` and `.jsx` files through Babel for transpilation before bundling them.
  * style-loader and css-loader, e.g., for `.css` file loading
  * file-loader or asset modules, e.g., for image file loading
3. Plugins, e.g., `HtmlWebpackPlugin` can automatically generate an index.html file
4. Optimization
  * Minification: Removing unnecessary characters from the code without changing its functionality to reduce file size.
  * Tree Shaking: Eliminating unused code from the final bundle.

## React Native

## `redux`

React component rendering is based on component states and props, on change the component and its child components will be re-rendered.
`redux` provides a custom solution to manually manage rendering.
`redux` is used to decouple this default rendering rule particularly useful for deeply nested components that have intricate dependencies to various components.

`redux` implements a centralized `store` that receives change signal from UI, rather than passively relying on component dependency chain changes.
The flexibility of Redux rendering control is a partnership between `dispatch` (the cause) and `useSelector` (the effect).

In the below example, 
the UI change signal is sent via `dispatch(incrementAction())` to redux `store`.
The component update (re-rendering) is queued and managed by React, that in this example when `value: state.value + 1` is updated via trigger `'INCREMENT'`, since the value has changed (`0 !== 1`), react-redux schedules a re-render with React.

```js
import { createStore } from 'redux';
import { Provider, useSelector, useDispatch } from 'react-redux';
import React from 'react';
import ReactDOM from 'react-dom';

// 1. Action: Define an action creator.
// This function creates an action object with a `type`.
const incrementAction = () => ({ type: 'INCREMENT' });

// 3. Reducer: A pure function to determine the new state.
// It takes the current state and an action, and returns the next state.
const counterReducer = (state = { value: 0 }, action) => {
  switch (action.type) {
    case 'INCREMENT':
      // 4. New State: Return a new state object.
      return { value: state.value + 1 };
    default:
      return state;
  }
};

// 2. Store: Create the central Redux store.
// The store holds the state and is created using the reducer.
const store = createStore(counterReducer);

// 6. Component Update: React component that connects to the Redux store.
function CounterComponent() {
  // Get the current state value from the store.
  const count = useSelector(state => state.value);
  // Get the dispatch function to send actions to the store.
  const dispatch = useDispatch();

  return (
    <div>
      <p>Count: {count}</p>
      {/* On click, an action is dispatched to the store, starting the cycle. */}
      <button onClick={() => dispatch(incrementAction())}>
        Increment
      </button>
    </div>
  );
}

// Render the application, wrapping it with the Provider to make the store available.
ReactDOM.render(
  <Provider store={store}>
    <CounterComponent />
  </Provider>,
  document.getElementById('root')
);

// 5. State Update: The store's state is now updated, and any component
// subscribed via `useSelector` will re-render with the new state.
```

## Responsive Programming

Responsive programming in react means applications adapt dynamically to different screen sizes, device types, and user interactions.

For example, React-Bootstrap or Material-UI come with built-in `grid` systems that are responsive by default.
In the below example, each column takes up the full width on small screens (`xs={12}`) but only half the width on medium screens (`md={6}`).

```js
import { Container, Row, Col } from 'react-bootstrap';

function ResponsiveGrid() {
  return (
    <Container>
      <Row>
        <Col xs={12} md={6}>
          Column 1
        </Col>
        <Col xs={12} md={6}>
          Column 2
        </Col>
      </Row>
    </Container>
  );
}
```


## StrictMode in React

`React.StrictMode` is a development-only wrapper component that activates additional checks and warnings for its descendant tree. It has **no effect in production**.

Enable it by wrapping your application (or a subtree) in `<React.StrictMode>`:

```jsx
import React from 'react';
import ReactDOM from 'react-dom/client';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

### What StrictMode Does

| Behavior | Purpose |
|---|---|
| **Double-invokes render functions** | Detects side effects in `render`, function component bodies, `useState`/`useMemo`/`useReducer` initializers |
| **Double-invokes effects** | Simulates mount → unmount → remount to surface missing cleanup in `useEffect` |
| **Warns on deprecated APIs** | Flags legacy lifecycle methods (e.g., `componentWillMount`) and other outdated patterns |
| **Warns on `findDOMNode`** | Discourages direct DOM node access via refs instead |

### Double-Invocation of Effects (React 18+)

In React 18, StrictMode deliberately runs each `useEffect` twice in development:

1. Mount → run effect
2. Unmount → run cleanup
3. Remount → run effect again

This exposes components that fail to clean up properly. A well-written effect must always pair its setup with a cleanup:

```jsx
useEffect(() => {
  const subscription = subscribe(resource);
  return () => subscription.unsubscribe(); // cleanup required
}, [resource]);
```

If the cleanup is missing or incorrect, the double-invocation will reveal bugs such as duplicate subscriptions or stale event listeners.

### `useRef` and StrictMode

`useRef` holds a mutable value in `.current` that **does not trigger re-renders** when mutated. Because it is mutable and persists across renders, its value is not reset between StrictMode's double-invocations — meaning ref-based side effects (e.g., click counters) may accumulate unexpected extra calls in development:

```jsx
function ClickCounter() {
  const clickCount = useRef(0);       // mutable, survives re-renders
  const [, forceRender] = useState(0);

  return (
    <>
      <button onClick={() => { clickCount.current++; }}>
        Click me
      </button>
      <button onClick={() => forceRender(n => n + 1)}>
        Force Re-Render
      </button>
      <p>Button clicked: {clickCount.current} times</p>
    </>
  );
}
```

The displayed count only updates on re-render (not on every click), because mutating `.current` does not schedule a render. In StrictMode, render is invoked twice, so render-time reads of `.current` may appear doubled in development — but this has no effect in production.

