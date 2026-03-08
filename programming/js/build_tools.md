# Development Tools

## `npm` and `npx`

### `npm` (Node Package Manager)

`npm` is the default package manager for Node.js, used to install, share, and manage dependencies in a project. It manages the `node_modules` folder and reads/writes to `package.json`.

**Common Commands & Options:**

*   **`npm init`**: Initializes a new project (creates `package.json`).
    *   `npm init -y`: Skips questions and uses defaults.
*   **`npm install <package>`** (or `npm i`): Installs a package.
    *   `--save-dev` / `-D`: Installs as a development dependency (e.g., test runners, build tools).
    *   `--global` / `-g`: Installs the package globally on the system (for CLI tools).
    *   `npm install` (no args): Installs all dependencies listed in `package.json`.
    *   `--production`: Skips installing `devDependencies` (useful for CI/CD).
*   **`npm run <script>`**: Runs a command defined in the `scripts` section of `package.json`.
*   **`npm update`**: Updates packages to the latest versions allowed by `package.json`.
*   **`npm list`**: Lists installed packages.
    *   `--depth=0`: Shows only top-level packages.

### `npx` (Node Package Execute)

`npx` is a package runner tool that comes bundled with npm (v5.2.0+).

**Key Features:**

1.  **Execute local binaries**: e.g., run `npx webpack` instead of `./node_modules/.bin/webpack`.
2.  **One-off execution**: Downloads and runs a package temporarily without installing it permanently. Great for scaffolding tools.
    *   Example: `npx create-react-app my-app`.
3.  **Run specific versions**: You can test a specific version of a library.
    *   Example: `npx node@14 index.js` (runs specific node version).

**Common Options:**

*   **`-p <package>`**: Specific package to install/use for the command.
    *   Example: `npx -p typescript tsc --init`.
*   **`--no-install`**: Fails if the package is not already installed locally (updates prevention).
*   **`--ignore-existing`**: Forces `npx` to ignore existing locally installed binaries and use a fresh cache.

## Webpack

```js
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  // 1. Entry point of the application
  entry: './src/index.js', 
  
  // 2. Output configuration where the bundled files will be placed
  output: {
    path: path.resolve(__dirname, 'dist'), 
    filename: 'bundle.js', 
    clean: true // Clean up the 'dist' folder before each build
  },
  
  // 3. Module rules to handle different file types
  module: {
    rules: [
      {
        // 4. Rule for JavaScript and JSX files (Transpile ES6+ and JSX to ES5)
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env', '@babel/preset-react'],
          },
        },
      },
      {
        // 5. Rule for CSS files
        test: /\.css$/,
        use: ['style-loader', 'css-loader'],
      },
      {
        // 6. Rule for images (handling images like .png, .jpg, etc.)
        test: /\.(png|svg|jpg|jpeg|gif)$/i,
        type: 'asset/resource',
      },
    ],
  },

  // 7. Plugins (extensions that hook into the Webpack build process)
  plugins: [
    new HtmlWebpackPlugin({
      template: './src/index.html', // Generates index.html and includes the bundle
    }),
  ],

  // 8. Development server configuration
  devServer: {
    contentBase: path.join(__dirname, 'dist'),
    compress: true, // Enable gzip compression for everything served
    port: 9000, // Serve the application at localhost:9000
    hot: true, // Enable hot module replacement
  },

  // 9. Set the mode for different optimizations (can be 'development', 'production', or 'none')
  mode: 'development',
};
```

### General Flow of Webpack

Webpack's build process follows a specific lifecycle:

1.  **Initialization**: Webpack reads the configuration file (`webpack.config.js`) and merges shell commands arguments.
2.  **Entry**: It starts looking for the entry point (e.g., `src/index.js`) to build a dependency graph.
3.  **Module Resolution & Transpilation**:
    *   It recursively finds all dependent modules (`import`/`require`).
    *   It applies **Loaders** (e.g., `babel-loader` for JS, `css-loader` for CSS) to transform non-JS files into valid modules.
4.  **Plugin Execution**: Throughout the process, **Plugins** hooks into specific lifecycle events (like `emit`, `compilation`) to perform tasks like code minification, asset generation, or environment variable injection.
5.  **Output**: Finally, it settles all modules and generates the bundled files into the output directory (e.g., `dist/bundle.js`).

### Plugins

While **Loaders** transform specific modules (per file type), **Plugins** serve a broader purpose. They can do anything that a loader cannot do. Plugins tap into the Webpack compilation lifecycle.

**Common Use Cases:**
*   **HtmlWebpackPlugin**: Generates an HTML file that automatically includes your hashed bundles.
*   **MiniCssExtractPlugin**: Extracts CSS into separate files (instead of internal style tags).
*   **DefinePlugin**: Defines global constants (like `process.env.NODE_ENV`) at compile time.
*   **CleanWebpackPlugin**: Cleans the build folder before each build.

### module federation

**Module Federation** is an advanced Webpack 5 feature that allows multiple separate builds (applications) to form a single application. This is the cornerstone technology for **Micro-Frontends**.

*   It allows a JavaScript application to dynamically load code from another application at **runtime**.
*   **Host**: The app consuming a module.
*   **Remote**: The app exposing a module.
*   It enables sharing common dependencies (like React or Lodash) so they aren't downloaded twice.

## Vite

Vite (pronounced "veet," the French word for "quick") is a high-performance equivalent to webpack.

Below is a typical example of `vite.config.js` file.

```js
/* eslint-env node */
/* eslint-disable no-undef */
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Load env file based on `mode` in the current working directory.
  // Set the third parameter to '' to load all env regardless of the `VITE_` prefix.
  const env = loadEnv(mode, process.cwd(), '')

  return {
    plugins: [react()],
    // Expose ENV variable to the client
    define: {
      'import.meta.env.ENV': JSON.stringify(env.ENV || 'dev')
    },
    server: {
      host: '0.0.0.0', // Allow external access
      port: 5173,      // Port number (change this to 80 or other port if needed)
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
          ws: true
        }
      }
    }
  }
})
```

### Env Setup

```js
export default defineConfig(({ mode }) => { 
  const env = loadEnv(mode, process.cwd(), '')
  ... 
  define: {
    'import.meta.env.ENV': JSON.stringify(env.ENV || 'dev')
  },
  ...
})
```

#### The `VITE_` Prefix

By default, Vite only exposes variables starting with `VITE_` to the client for security.
For example, imported env vars need to have a prefix of `VITE_`, otherwise they are not recognized parsed as `undefined`.

```env
VITE_API_URL=https://api.example.com
VITE_APP_TITLE="My App"
```

To use them in react, there is

```jsx
function App() {
  const apiUrl = import.meta.env.VITE_API_URL;
  const mode = import.meta.env.MODE;

  return (
    <div>
      <h1>App running in {mode} mode</h1>
      <p>API URL: {apiUrl}</p>
    </div>
  );
}
```

By passing an empty string `''` as the third argument to `loadEnv`, aLL environment variables from `.env` file are loaded into the env constant, regardless of their prefix.

#### Env Var Loading

The code needs to know the mode to load the specific `.env` file associated with it.

For example, there are `.env.prod`, `.env.preprod`, `.env.uat`, `.env.dev` env files.
vite by the below `.env` to use the corresponding env file (e.g., to use `prod` env).

```env
VITE_ENV=prod
```

`import.meta.env.ENV` is used to pass env vars to react jsx code.

|Var|Description|Default Value|
|:---|:---|:---|
|`import.meta.env.MODE`|The current mode (`dev`, `prod`, etc.)|`'dev'`|
|`import.meta.env.BASE_URL`|The base URL of your app (from base config)|`'/'`|


### The Proxy Host

```js
server: {
  host: '0.0.0.0', // Allow external access
  port: 5173,      // Port number (change this to 80 or other port if needed)
  proxy: {
    '/api': {
      target: 'http://127.0.0.1:8000',
      changeOrigin: true,
      ws: true
    }
  }
}
```

The `5173` is the ui server. This server in development (NOT used on production) can do hot update on UI rendering for any new code change.
The actual code on prod will be hosted on `8000`.

When the `8000` backend replies, Vite passes the data back to React via `5173`.
* `changeOrigin: true`: Modifies the "Origin" header of the request to match the target url (tricks the backend into thinking the request is coming from port `8000`, not `5173`).
* `ws: true`: Enables WebSocket proxying (useful if backend uses real-time sockets).

### Vite Build Process

Just need to run `npm run build` to build the monolith `dist` folder.

```txt
dist/
├── index.html              <-- The ENTRY POINT. (This is the only HTML file)
├── favicon.ico
└── assets/
    ├── index-D8s7f9a.js    <-- The entire React App bundled into one (or few) files
    └── index-B2d1s5e.css   <-- All your CSS styles
```

For backend (e.g., fastAPI) to know this `dist` folder, need to hook up the folders between `dist` vs `/assets`.

```py
if os.path.exists(dist_path):
    # Only mount assets if the directory exists
    assets_path = os.path.join(dist_path, "assets")
    if os.path.exists(assets_path):
        app.mount(
            "/assets",
            StaticFiles(directory=assets_path),
            name="assets",
        )
        logger.info(f"Serving assets from: {assets_path}")
    else:
        logger.warning(f"Warning: {assets_path} not found. Assets will not be served.")
```
