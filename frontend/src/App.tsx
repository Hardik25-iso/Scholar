import { Navigate, Route, Routes } from "react-router-dom";
import { useAuth } from "./auth/AuthContext";
import ProtectedRoute from "./auth/ProtectedRoute";
import AuthForm from "./pages/AuthForm";
import Landing from "./pages/Landing";
import Workspace from "./pages/Workspace";

/** Public routes bounce to /app when already authenticated. */
function PublicOnly({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) return null;
  return user ? <Navigate to="/app" replace /> : <>{children}</>;
}

export default function App() {
  return (
    <Routes>
      {/* Public landing page — the front door, accessible signed in or out. */}
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<PublicOnly><AuthForm mode="login" /></PublicOnly>} />
      <Route path="/signup" element={<PublicOnly><AuthForm mode="signup" /></PublicOnly>} />

      {/* Protected app */}
      <Route element={<ProtectedRoute />}>
        <Route path="/app" element={<Workspace />} />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
