"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useStore } from "@/lib/store";

const LINKS = [
  { href: "/run", label: "Run" },
  { href: "/trajectories", label: "Trajectories" },
  { href: "/compare", label: "Compare" },
  { href: "/trend", label: "Trend" },
];

export function NavBar() {
  const pathname = usePathname();
  const { primary, clear } = useStore();

  return (
    <header className="border-b border-neutral-200 dark:border-neutral-800">
      <div className="mx-auto flex max-w-6xl items-center gap-6 px-4 py-3">
        <Link href="/" className="font-mono text-sm font-semibold tracking-tight">
          trajeval
        </Link>
        {primary && (
          <nav className="flex gap-4 text-sm">
            {LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={
                  pathname.startsWith(link.href)
                    ? "font-medium text-neutral-900 dark:text-neutral-100"
                    : "text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100"
                }
              >
                {link.label}
              </Link>
            ))}
          </nav>
        )}
        <div className="flex-1" />
        {primary && (
          <button
            type="button"
            onClick={clear}
            className="text-xs text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100"
          >
            clear loaded run
          </button>
        )}
      </div>
    </header>
  );
}
