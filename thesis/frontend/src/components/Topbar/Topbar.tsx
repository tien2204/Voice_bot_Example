"use client";

import { useAuth } from "@/contexts/AuthContext";
import Image from "next/image";
import Link from "next/link"
import { useRouter, usePathname } from "next/navigation";
import React, { useEffect } from "react";
import { SparkleDiv } from "../Sparkle/SparkleDiv";

const Topbar: React.FC = () => {
  const { user, isAuthenticated, logout, isLoading, fetchCurrentUser } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const [isDropdownOpen, setIsDropdownOpen] = React.useState(false);
  const dropdownRef = React.useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!user && !isLoading) {
      fetchCurrentUser().then((val) => {

        if (!val) {
          logout()
          router.push("/login");
        }
      });
    }
  }, [user, isLoading]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);


  const handleLogout = async () => {
    logout();
    router.push("/login");
  };

  const getLinkClass = (href: string) => {
    const baseStyle = "py-2 px-3 rounded-md text-white text-base";
    return pathname === href
      ? `${baseStyle} font-bold pointer-events-none text-gray-400` // Style for active/disabled link
      : `${baseStyle} hover:bg-gray-700 hover:text-sky-300`; // Style for normal link
  };
  const isLinkActive = (href: string) => {
    return pathname === href;
  }

  return (
    <nav className="bg-gray-800 text-white py-1 px-6 flex justify-between items-center shadow flex-shrink-0">
      <Link href="/">
        <div className="flex flex-row justify-center items-center text-white text-2xl">
          <Image src="/favicon.png" alt="VoiceApp Logo" width={60} height={60} />
          <span className="text-sky-400 text-2xl font-extrabold pt-3 ml-2"> {/* Added ml-2 for spacing, adjust as needed */}
            {process.env.NEXT_PUBLIC_APP_NAME}
          </span>
        </div>
      </Link>

      <div className="flex items-center gap-4">
        {isLoading ? (
          <span className="text-base">Loading...</span>
        ) : isAuthenticated && user ? (
          <>
            {isLinkActive("/avatar/create") ? (
              <span className={getLinkClass("/avatar/create")}>Create Avatar</span>
            ) : (
              <Link href="/avatar/create" className={getLinkClass("/avatar/create")}>
                {/* Apply SparkleDiv if you want sparkles on this link */}
                {/* <SparkleDiv>Create Avatar</SparkleDiv> */}
                Create Avatar
              </Link>
            )}
            {isLinkActive("/avatar/chat") ? (
              <span className={getLinkClass("/avatar/chat")}>Chat with Avatar</span>
            ) : (
              <Link href="/avatar/chat" className={getLinkClass("/avatar/chat")}>
                Chat with Avatar
              </Link>
            )}
            <div className="relative" ref={dropdownRef}>
              <button
                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                className="text-white text-base py-2 px-3 rounded-md hover:bg-gray-700 focus:outline-none"
              >
                {user.username}
                <svg className={`inline-block w-4 h-4 ml-1 transition-transform duration-200 ${isDropdownOpen ? 'transform rotate-180' : ''}`} fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd"></path></svg>
              </button>
              {isDropdownOpen && (
                <div className="absolute right-0 mt-2 bg-gray-700 rounded-md shadow-lg py-1 z-50">
                  <div className="px-4 py-2 text-sm text-gray-300">{user.email}</div>
                  <button
                    onClick={handleLogout}
                    className="block w-full text-left px-4 py-2 text-sm text-white hover:bg-gray-600"
                  >
                    Logout
                  </button>
                </div>
              )}
            </div>
          </>
        ) : (
          <>{/* <Link href="/login">Login</Link> */}</>
        )}
      </div>
    </nav>
  );
};

export default Topbar;
