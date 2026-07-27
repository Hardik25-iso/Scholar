import { Navigate, Outlet } from "react-router-dom";
import { useAuth } from "./AuthContext";

/** Gate for authenticated routes: redirect to /login until /me confirms a user. */
export default function ProtectedRoute() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-paper">
        <span className="font-mono text-xs uppercase tracking-[0.18em] text-faint">
          checking session…
        </span>
      </div>
    );
  }
  return user ? <Outlet /> : <Navigate to="/login" replace />;
}
