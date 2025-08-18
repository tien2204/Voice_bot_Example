"use client";

import Link from "next/link";
import React from "react";

const Footbar: React.FC = () => {
  const currentYear = new Date().getFullYear();
  const companyName = process.env.NEXT_PUBLIC_APP_NAME || "Company";
  const contactEmail = process.env.NEXT_PUBLIC_EMAIL || "Hotline email"; // Replace with your actual email
  const githubLink = process.env.NEXT_PUBLIC_GITHUB_URL || "/"; // Replace with your actual GitHub link

  return (
    <footer className="bg-gray-800 text-gray-400 p-4 text-sm">
      <div className="container mx-auto flex justify-between items-center">
        <p>
          &copy; {currentYear} {companyName}. All rights reserved.
        </p>
        <p>
          Contact us:{" "}
          <a href={`mailto:${contactEmail}`} className="hover:text-sky-400">
            {contactEmail}
          </a>
          {" | "}
          <Link href={githubLink} target="_blank" rel="noopener noreferrer" className="hover:text-sky-400">
            GitHub
          </Link>
        </p>
      </div>
    </footer>
  );
};

export default Footbar;
