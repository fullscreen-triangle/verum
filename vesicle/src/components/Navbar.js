import Link from "next/link";
import React, { useState } from "react";
import Logo from "./Logo";
import { useRouter } from "next/router";
import { GithubIcon, LinkedInIcon } from "./Icons";
import { motion, AnimatePresence } from "framer-motion";

// Grouped navigation structure
const NAV_GROUPS = [
  {
    label: "Technology",
    items: [
      { href: "/platform", title: "Platform" },
      { href: "/membrane", title: "Membrane" },
      { href: "/oscillations", title: "Oscillations" },
      { href: "/semiconductor", title: "Semiconductor" },
      { href: "/systems", title: "Systems" },
    ],
  },
  {
    label: "Diagnostics",
    items: [
      { href: "/philharmonic", title: "Philharmonic" },
      { href: "/f1", title: "F1 Bahrain" },
      { href: "/dashboard", title: "Dashboard" },
    ],
  },
  {
    label: "Navigation",
    items: [
      { href: "/navigation", title: "Navigation" },
      { href: "/network", title: "Network" },
      { href: "/weather", title: "Weather" },
    ],
  },
];

const TOP_LINKS = [
  { href: "/", title: "Home" },
  { href: "/papers", title: "Papers" },
  { href: "/invest", title: "Invest" },
];

function NavDropdown({ label, items }) {
  const [open, setOpen] = useState(false);
  const router = useRouter();
  const isActive = items.some((item) => router.asPath === item.href);

  return (
    <div
      className="relative mx-3"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        className={`text-sm font-medium transition-colors ${
          isActive ? "text-gold" : "text-light/80 hover:text-light"
        }`}
      >
        {label}
        <span className="ml-1 text-[10px] opacity-50">▾</span>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            transition={{ duration: 0.15 }}
            className="absolute top-full left-0 mt-2 min-w-[180px] bg-dark/95 backdrop-blur-md border border-light/10 rounded-lg py-2 shadow-xl z-50"
          >
            {items.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={`block px-4 py-2 text-sm transition-colors ${
                  router.asPath === item.href
                    ? "text-gold bg-gold/5"
                    : "text-light/70 hover:text-light hover:bg-light/5"
                }`}
              >
                {item.title}
              </Link>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function TopLink({ href, title }) {
  const router = useRouter();
  return (
    <Link
      href={href}
      className={`mx-3 text-sm font-medium transition-colors relative group ${
        router.asPath === href ? "text-gold" : "text-light/80 hover:text-light"
      }`}
    >
      {title}
      <span
        className={`inline-block h-[1px] bg-gold absolute left-0 -bottom-0.5 group-hover:w-full transition-[width] ease duration-300 ${
          router.asPath === href ? "w-full" : "w-0"
        }`}
      >
        &nbsp;
      </span>
    </Link>
  );
}

// All links flattened for mobile
const ALL_MOBILE_LINKS = [
  { href: "/", title: "Home" },
  { href: "/platform", title: "Platform" },
  { href: "/membrane", title: "Membrane" },
  { href: "/oscillations", title: "Oscillations" },
  { href: "/semiconductor", title: "Semiconductor" },
  { href: "/systems", title: "Systems" },
  { href: "/philharmonic", title: "Philharmonic" },
  { href: "/f1", title: "F1 Bahrain" },
  { href: "/dashboard", title: "Dashboard" },
  { href: "/navigation", title: "Navigation" },
  { href: "/network", title: "Network" },
  { href: "/weather", title: "Weather" },
  { href: "/papers", title: "Papers" },
  { href: "/invest", title: "Invest" },
];

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const router = useRouter();

  const handleClick = () => setIsOpen(!isOpen);

  return (
    <header className="w-full flex items-center justify-between px-32 py-6 font-medium z-10 text-light lg:px-16 relative md:px-12 sm:px-8">
      {/* Mobile hamburger */}
      <button
        type="button"
        className="flex-col items-center justify-center hidden lg:flex"
        onClick={handleClick}
      >
        <span className="sr-only">Open menu</span>
        <span className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ease-out ${isOpen ? "rotate-45 translate-y-1" : "-translate-y-0.5"}`} />
        <span className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ease-out ${isOpen ? "opacity-0" : "opacity-100"} my-0.5`} />
        <span className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ease-out ${isOpen ? "-rotate-45 -translate-y-1" : "translate-y-0.5"}`} />
      </button>

      {/* Desktop nav */}
      <div className="w-full flex justify-between items-center lg:hidden">
        <nav className="flex items-center">
          <TopLink href="/" title="Home" />

          {NAV_GROUPS.map((group) => (
            <NavDropdown key={group.label} label={group.label} items={group.items} />
          ))}

          <TopLink href="/papers" title="Papers" />
          <TopLink href="/invest" title="Invest" />
        </nav>

        <nav className="flex items-center gap-3">
          <motion.a
            target="_blank"
            className="w-6 bg-light rounded-full"
            href="https://github.com/fullscreen-triangle"
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.9 }}
          >
            <GithubIcon />
          </motion.a>
          <motion.a
            target="_blank"
            className="w-6"
            href="#"
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.9 }}
          >
            <LinkedInIcon />
          </motion.a>
        </nav>
      </div>

      {/* Mobile menu */}
      {isOpen && (
        <motion.div
          className="min-w-[70vw] sm:min-w-[90vw] flex justify-between items-center flex-col fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 py-16 bg-dark/95 rounded-lg z-50 backdrop-blur-md border border-light/10"
          initial={{ scale: 0, x: "-50%", y: "-50%", opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
        >
          <nav className="flex items-center justify-center flex-col gap-1">
            {ALL_MOBILE_LINKS.map((link) => (
              <button
                key={link.href}
                className={`text-sm py-1.5 px-4 rounded transition-colors ${
                  router.asPath === link.href ? "text-gold" : "text-light/70"
                }`}
                onClick={() => {
                  handleClick();
                  router.push(link.href);
                }}
              >
                {link.title}
              </button>
            ))}
          </nav>
          <nav className="flex items-center gap-4 mt-6">
            <motion.a target="_blank" className="w-6 bg-light rounded-full" href="https://github.com/fullscreen-triangle" whileHover={{ y: -2 }}>
              <GithubIcon />
            </motion.a>
            <motion.a target="_blank" className="w-6" href="#" whileHover={{ y: -2 }}>
              <LinkedInIcon />
            </motion.a>
          </nav>
        </motion.div>
      )}

      <div className="absolute left-[50%] top-2 translate-x-[-50%]">
        <Logo />
      </div>
    </header>
  );
};

export default Navbar;
