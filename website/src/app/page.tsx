import { DemoBlock } from "@/components/DemoBlock";
import { Explore } from "@/components/Explore";
import { Footer } from "@/components/Footer";
import { Hero } from "@/components/Hero";
import { Problem } from "@/components/Problem";
import { Results } from "@/components/Results";
import { WhatWeBuilt } from "@/components/WhatWeBuilt";

export default function Home() {
  return (
    <main>
      <Hero />
      <DemoBlock />
      <Problem />
      <WhatWeBuilt />
      <Results />
      <Explore />
      <Footer />
    </main>
  );
}
