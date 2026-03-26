import { motion } from "framer-motion";

export default function SectionHeading({ title, subtitle, className = "" }) {
  return (
    <motion.div
      initial={{ y: 30, opacity: 0 }}
      whileInView={{ y: 0, opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
      className={`text-center mb-12 ${className}`}
    >
      <h2 className="font-bold text-5xl mb-4 text-light md:text-4xl sm:text-3xl">
        {title}
      </h2>
      {subtitle && (
        <p className="text-lg text-light/60 max-w-3xl mx-auto">{subtitle}</p>
      )}
    </motion.div>
  );
}
