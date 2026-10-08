// The desktop renderer is a plain Vite SPA, so the workspace page's single
// Next-only import ("next/link") resolves here via vite.renderer.config.ts.
// The brand link points at "/" and the page has no routing; a styled anchor is
// behaviorally identical, and external links are opened by the main process.
import { forwardRef, type AnchorHTMLAttributes } from "react";

const Link = forwardRef<HTMLAnchorElement, AnchorHTMLAttributes<HTMLAnchorElement>>(
  function Link({ children, ...props }, ref) {
    return <a ref={ref} {...props}>{children}</a>;
  },
);

export default Link;
