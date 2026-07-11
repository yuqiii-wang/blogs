# React Rendering

## General React Rendering Rules

### Parent Component Changer Cascading Effect on Child Components

### Deep vs. Shallow Comparison

Re-render decision often boils down to comparing the component's previous props and state with the new ones. The two primary methods for this comparison are "shallow comparison" and "deep comparison."

Different hook methods implement different comparison algos.

#### Shallow Comparison

* For primitive types (like numbers, strings, booleans): A shallow comparison checks if the **values** are identical. For example, 5 is equal to 5, and 'hello' is equal to 'hello'.
* For complex types (like objects and arrays): A shallow comparison checks if the **references** (the location in memory) are the same. It does not check the individual elements or properties within the object or array.

#### Deep Comparison

* It traverses the entire structure of an object or array and compares the values of each nested property or element, based on which comparison is performed.

## Event Bubbling

Below explains how to control preventing a child component from propagating event up to a parent component by `stopPropagation()`.

A click event travels in three phases: **capture** (down from `window`), **target** (the clicked element), then **bubble** (back up to `window`). React's `onClick` listens in the bubble phase, so a single click can fire handlers on the target *and* all its ancestors.

### Problem: Unintended Row Navigation

```jsx
<Table onRow={() => ({ onClick: () => navigate(threadId) })}>   {/* ← 3. fires navigate() */}
  <tr>                                                           {/* ← 2. bubbles up */}
    <td>
      <span onClick={() => setExpanded(true)}>…</span>          {/* ← 1. you click here */}
    </td>
  </tr>
</Table>
```

Clicking the `<span>` sets expanded state **and** bubbles up to the row handler, triggering navigation — both fire unintentionally.

### Fix: `e.stopPropagation()`

Stops the event from travelling further up the DOM; ancestors never see it.

```jsx
onClick={(e) => {
  e.stopPropagation(); // event stops here — row handler never fires
  setExpanded(true);
}}
```

> **`stopPropagation` vs `preventDefault`**
> - `stopPropagation()` — blocks bubbling, does **not** cancel browser defaults (form submit, link click).
> - `preventDefault()` — cancels browser defaults, does **not** stop bubbling.

### Blanket Guard on a Container

Instead of adding `stopPropagation` to every child, put one guard on the outermost container to make it a **click boundary**:

```jsx
<div onClick={(e) => e.stopPropagation()}>
  {/* nothing inside can bubble out */}
</div>
```

Use this only when you intentionally want to block all outward click propagation from the component.