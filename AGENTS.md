# AGENTS.md

This document provides guidelines for AI agents working on this codebase.

## Build, Lint, and Test Commands

### Core Commands
```bash
# Development server with hot reload
npm run dev

# Build for production
npm run build

# Run type checker and linter
npm run lint

# Preview production build locally
npm run preview
```

### TypeScript
- Type checking is enforced via `tsc -b` in the build command
- Strict mode is enabled in `tsconfig.app.json`
- No unused locals or parameters allowed

### ESLint
- Linting is configured in `eslint.config.js`
- Extends: ESLint recommended, TypeScript-ESLint recommended, React Hooks, React Refresh
- Ignores `dist` directory
- Run with `npm run lint` or `eslint .`

### Testing
- **No test framework is currently installed**
- To add tests, install a framework (e.g., Vitest, Jest) and update package.json scripts
- When tests are added, run a single test file using your test runner's specific command

## Code Style Guidelines

### TypeScript
- Use TypeScript for all files (`.ts` for utilities, `.tsx` for React components)
- Enable `strict: true` in TypeScript configuration
- Use explicit types for function parameters and return values
- Use `interface` for object types, `type` for unions, primitives, and utility types
- Avoid `any` - use `unknown` when type is truly unknown
- Use `Map` for dictionary-like structures (see `useQuiz.ts:31`)
- Use array methods (`map`, `filter`, `reduce`, `forEach`) for transformations
- Prefer `Array.from()` for array conversions (see `useQuiz.ts:161`)
- Use early returns to reduce nesting and improve readability

### React Components
- Use functional components with TypeScript
- Type props explicitly using `interface` or `type` (e.g., `QuestionCardProps`)
- Use `React.FC<PropsType>` for component type annotations
- Component files: PascalCase (e.g., `QuestionCard.tsx`)
- Place component in `src/components/` organized by feature
- Default export components
- Extract UI constants outside components (see `QuestionCard.tsx:18-32`)
- Use `useState` for component-local state management
- Use `useEffect` with proper cleanup (removeEventListener)
- Include responsive breakpoints with `window.matchMedia`

### Naming Conventions
- **Files**: PascalCase for components, camelCase for utilities/hooks
- **Variables/functions**: camelCase (e.g., `currentQuestion`, `toggleOption`)
- **Constants**: UPPER_SNAKE_CASE or `as const` object patterns (see `src/types/quiz.ts`)
- **Interfaces/Types**: PascalCase (e.g., `Question`, `QuizResults`)
- **Enums**: Use `const` objects with `as const` and derived types:
  ```typescript
  export const QuestionType = {
    MCQ: 'mcq',
    MSQ: 'msq',
  } as const;
  export type QuestionType = typeof QuestionType[keyof typeof QuestionType];
  ```

### Imports
- React: Named imports from `react` (e.g., `useState, useEffect`)
- Component imports: Named imports for components
- Type imports: Use `import type` for type-only imports
- Organize imports in this order:
  1. React imports
  2. Third-party library imports
  3. Internal imports (components, hooks, types)
  4. CSS/style imports

### Hooks
- Custom hooks: Prefix with `use`, placed in `src/hooks/`
- Use `useCallback` for functions passed as props
- Use `useMemo` for expensive computations
- Include JSDoc comments for hook purpose and parameters

### CSS and Styling
- This project uses Tailwind CSS v4 via `@tailwindcss/vite`
- Use utility classes for styling
- Inline styles are used for dynamic values (media queries, responsive breakpoints)
- Place global styles in `src/index.css`, component styles in `src/App.css`

### Error Handling
- Handle null/undefined cases explicitly
- Use optional chaining (`?.`) and nullish coalescing (`??`)
- Validate props with default values
- Return early for error conditions
- Always guard against accessing properties on potentially null/undefined values

### Data Structures
- Use `Map` for key-value collections that need frequent insertion/deletion (see `useQuiz.ts`)
- Use arrays for ordered collections
- Use `Set` for unique value collections
- Convert Maps to arrays using `Array.from()` when iteration is needed
- Clone Maps using `new Map(originalMap)` to avoid mutation

### Project Structure
```
src/
├── components/
│   └── quiz/          # Quiz-related components
├── hooks/
│   └── useQuiz.ts     # Custom quiz hook
├── types/
│   └── quiz.ts        # TypeScript type definitions
├── App.tsx            # Main app component
├── main.tsx           # Entry point
└── index.css          # Global styles
```

### Miscellaneous
- Use semicolons in JavaScript/TypeScript
- Prefer arrow functions for callbacks and closures
- Use template literals for string interpolation
- Avoid magic numbers - use named constants
- ESLint will enforce most formatting rules
