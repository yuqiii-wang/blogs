# Common Native React Components


## React Router

React provides two router component types:

|`<Route>`|`<Link>`|
|:---|:---|
|Declares which **component to render** for a specific URL path.|	Creates a **clickable element** (an `<a>` tag) that allows users to navigate to a different URL path.|

In the below example, define `<Home />` and `<About />` to render matched against what routing path.

```js
import React from 'react';
import { Routes, Route } from 'react-router-dom'; // Import Routes and Route
import Navbar from './components/Navbar';
import Home from './components/Home';
import About from './components/About';

function App() {
  return (
    <div>
      <Navbar />

      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
```

`<Link />` is used in navigation bar served as an `<a>` to click.

```js
import React from 'react';
import { Link } from 'react-router-dom'; // Import Link

function Navbar() {
  return (
    <nav style={{ marginBottom: '20px', borderBottom: '1px solid #ccc', paddingBottom: '10px' }}>
      <Link to="/" style={{ marginRight: '15px' }}>Home</Link>
      <Link to="/about">About</Link>
    </nav>
  );
}

export default Navbar;
```